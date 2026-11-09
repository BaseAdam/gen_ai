import logging
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import httpx
import psycopg
import pytest
from fastapi.testclient import TestClient
from google.genai import errors, types
from pydantic import SecretStr

import offers.api
from offers.api import app, get_client, get_conn, get_offers, get_settings
from offers.config import Settings
from offers.embeddings import EMBEDDING_DIMENSIONS
from offers.models import Offer
from offers.observability import current_request
from offers.store import Connection, replace_offers

OFFERS = [
    Offer(title="A", company="X", city="Warsaw", salary_min=10000, salary_max=20000),
    Offer(title="B", company="Y", city="Krakow", remote=True, skills=["python"]),
]

CHUNKS = [["a"], ["b"]]


def vec(*head: float) -> list[float]:
    return list(head) + [0.0] * (EMBEDDING_DIMENSIONS - len(head))


def text_response(text: str) -> types.GenerateContentResponse:
    content = types.Content(role="model", parts=[types.Part(text=text)])
    return types.GenerateContentResponse(candidates=[types.Candidate(content=content)])


def call_response(name: str, **args: Any) -> types.GenerateContentResponse:
    part = types.Part(function_call=types.FunctionCall(name=name, args=args))
    content = types.Content(role="model", parts=[part])
    return types.GenerateContentResponse(candidates=[types.Candidate(content=content)])


class FakeModels:
    def __init__(self) -> None:
        self.responses: list[types.GenerateContentResponse] = []
        self.chunks: list[str] = []
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None

    def generate_content(self, **kwargs: Any) -> types.GenerateContentResponse:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.responses.pop(0)

    def generate_content_stream(self, **kwargs: Any) -> Iterator[SimpleNamespace]:
        self.calls.append(kwargs)
        for text in self.chunks:
            yield SimpleNamespace(text=text)

    def embed_content(self, **kwargs: Any) -> types.EmbedContentResponse:
        self.calls.append(kwargs)
        embedding = types.ContentEmbedding(values=vec(1.0, 0.0))
        return types.EmbedContentResponse(embeddings=[embedding])


class FakeClient:
    def __init__(self) -> None:
        self.models = FakeModels()


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def api(fake_client: FakeClient) -> Iterator[TestClient]:
    settings = Settings(
        gemini_api_key=SecretStr("test-key"),
        gemini_model="test-model",
        rag_min_score=0.5,
        rag_rerank=False,
        _env_file=None,
    )
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_client] = lambda: fake_client
    app.dependency_overrides[get_offers] = lambda: OFFERS
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_lists_all_offers(api: TestClient) -> None:
    response = api.get("/offers")

    assert response.status_code == 200
    assert [offer["title"] for offer in response.json()] == ["A", "B"]


def test_filters_offers_by_query_params(api: TestClient) -> None:
    response = api.get("/offers", params={"city": "krakow", "remote": "true"})

    assert [offer["title"] for offer in response.json()] == ["B"]


def test_extract_returns_offer(api: TestClient, fake_client: FakeClient) -> None:
    raw = '{"title": "Dev", "company": "Acme", "city": "Gdansk", "skills": ["Python"]}'
    fake_client.models.responses = [text_response(raw)]

    response = api.post("/offers/extract", json={"text": "Hiring a Python dev at Acme in Gdansk"})

    assert response.status_code == 200
    assert response.json()["company"] == "Acme"
    assert response.json()["skills"] == ["python"]
    assert fake_client.models.calls[0]["model"] == "test-model"


def test_extract_rejects_short_text_without_calling_model(
    api: TestClient, fake_client: FakeClient
) -> None:
    response = api.post("/offers/extract", json={"text": "too short"})

    assert response.status_code == 422
    assert fake_client.models.calls == []


def test_ask_runs_tool_and_returns_answer(api: TestClient, fake_client: FakeClient) -> None:
    fake_client.models.responses = [
        call_response("salary_stats", city="Warsaw"),
        text_response("On average 15000 PLN"),
    ]

    response = api.post("/ask", json={"question": "How much do they pay in Warsaw?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "On average 15000 PLN"}
    assert len(fake_client.models.calls) == 2


def test_tips_streams_text(api: TestClient, fake_client: FakeClient) -> None:
    fake_client.models.chunks = ["Review ", "", "asyncio."]

    response = api.get("/offers/0/tips")

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    assert response.text == "Review asyncio."


@pytest.mark.parametrize("index", [2, -1])
def test_tips_unknown_offer_returns_404(
    api: TestClient, fake_client: FakeClient, index: int
) -> None:
    response = api.get(f"/offers/{index}/tips")

    assert response.status_code == 404
    assert response.json() == {"detail": "Offer not found"}
    assert fake_client.models.calls == []


def test_model_timeout_returns_504(api: TestClient, fake_client: FakeClient) -> None:
    fake_client.models.error = httpx.ReadTimeout("too slow")

    response = api.post("/ask", json={"question": "How much do they pay in Warsaw?"})

    assert response.status_code == 504
    assert response.json() == {"detail": "The model did not respond in time"}


def test_rag_uses_reranker_when_enabled(
    api: TestClient, fake_client: FakeClient, conn: Connection
) -> None:
    settings = Settings(
        gemini_api_key=SecretStr("test-key"),
        gemini_model="test-model",
        rag_min_score=0.5,
        rag_rerank=True,
        _env_file=None,
    )
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_conn] = lambda: conn
    replace_offers(conn, OFFERS, CHUNKS, [[vec(1.0, 1.0)], [vec(1.0, 0.0)]])
    relevance = '{"scores": [{"index": 1, "score": 2}, {"index": 2, "score": 8}]}'
    fake_client.models.responses = [text_response(relevance), text_response("I recommend [1].")]

    response = api.post("/rag", json={"question": "Best pick?"})

    body = response.json()
    assert body["answer"] == "I recommend [1]."
    assert [source["offer"]["title"] for source in body["sources"]] == ["A"]


def test_model_error_returns_502_without_details(api: TestClient, fake_client: FakeClient) -> None:
    body = {"error": {"code": 503, "message": "internal detail", "status": "UNAVAILABLE"}}
    fake_client.models.error = errors.APIError(503, body)

    response = api.post("/offers/extract", json={"text": "Hiring a Python dev at Acme in Gdansk"})

    assert response.status_code == 502
    assert "internal detail" not in response.text


def test_rag_returns_answer_with_sources(
    api: TestClient, fake_client: FakeClient, conn: Connection
) -> None:
    app.dependency_overrides[get_conn] = lambda: conn
    replace_offers(conn, OFFERS, CHUNKS, [[vec(0.0, 1.0)], [vec(1.0, 0.0)]])
    fake_client.models.responses = [text_response("I recommend [1].")]

    response = api.post("/rag", json={"question": "Best pick?", "top_k": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "I recommend [1]."
    assert [source["offer"]["title"] for source in body["sources"]] == ["B"]
    assert body["sources"][0]["score"] == pytest.approx(1 / 61)


def test_rag_returns_503_when_database_is_down(api: TestClient, fake_client: FakeClient) -> None:
    def broken_conn() -> Connection:
        raise psycopg.OperationalError("connection refused")

    app.dependency_overrides[get_conn] = broken_conn

    response = api.post("/rag", json={"question": "Best pick?"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Database is unavailable"}
    assert fake_client.models.calls == []


def test_rag_passes_filters_from_request(
    api: TestClient, fake_client: FakeClient, conn: Connection
) -> None:
    app.dependency_overrides[get_conn] = lambda: conn
    replace_offers(conn, OFFERS, CHUNKS, [[vec(1.0, 1.0)], [vec(1.0, 0.0)]])
    fake_client.models.responses = [text_response("I recommend [1].")]

    response = api.post("/rag", json={"question": "Best pick?", "remote": False})

    assert [source["offer"]["title"] for source in response.json()["sources"]] == ["A"]


def test_rag_uses_min_score_from_settings(
    api: TestClient, fake_client: FakeClient, conn: Connection
) -> None:
    app.dependency_overrides[get_conn] = lambda: conn
    replace_offers(conn, OFFERS, CHUNKS, [[vec(0.0, 1.0)], [vec(1.0, 0.0)]])
    fake_client.models.responses = [text_response("I recommend [1].")]

    response = api.post("/rag", json={"question": "Best pick?"})

    assert [source["offer"]["title"] for source in response.json()["sources"]] == ["B"]


def test_every_response_gets_its_own_request_id(api: TestClient) -> None:
    first = api.get("/offers").headers["X-Request-ID"]
    second = api.get("/offers").headers["X-Request-ID"]

    assert len(first) == 8
    assert first != second


def test_request_id_is_visible_inside_endpoint(
    api: TestClient, fake_client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = '{"title": "Dev", "company": "Acme", "city": "Gdansk"}'
    fake_client.models.responses = [text_response(raw)]
    seen: list[str] = []
    original = fake_client.models.generate_content

    def spy(**kwargs: Any) -> types.GenerateContentResponse:
        context = current_request.get()
        seen.append(context.request_id if context else "-")
        return original(**kwargs)

    monkeypatch.setattr(fake_client.models, "generate_content", spy)

    response = api.post("/offers/extract", json={"text": "Hiring a Python dev at Acme in Gdansk"})

    assert seen == [response.headers["X-Request-ID"]]


def test_request_summary_is_logged(api: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="offers"):
        api.get("/offers/99/tips")

    [summary] = [record for record in caplog.records if record.name == "offers.api"]
    assert summary.getMessage().startswith("GET /offers/99/tips -> 404 in ")
    assert (summary.method, summary.path, summary.status) == ("GET", "/offers/99/tips", 404)


def test_request_summary_includes_tokens_used_by_endpoint(
    api: TestClient, fake_client: FakeClient, conn: Connection, caplog: pytest.LogCaptureFixture
) -> None:
    app.dependency_overrides[get_conn] = lambda: conn
    replace_offers(conn, OFFERS, CHUNKS, [[vec(0.0, 1.0)], [vec(1.0, 0.0)]])
    reply = text_response("I recommend [1].")
    reply.usage_metadata = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=700, candidates_token_count=100, thoughts_token_count=20
    )
    fake_client.models.responses = [reply]

    with caplog.at_level(logging.INFO, logger="offers"):
        api.post("/rag", json={"question": "Best pick?"})

    [summary] = [message for message in caplog.messages if message.startswith("POST /rag")]
    assert summary.endswith(", tokens: input=700 output=100 thinking=20")


def test_startup_configures_logging_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        gemini_api_key=SecretStr("test-key"),
        gemini_model="test-model",
        log_json=True,
        _env_file=None,
    )
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(offers.api, "get_settings", lambda: settings)
    monkeypatch.setattr(offers.api, "configure_logging", lambda **kwargs: calls.append(kwargs))

    with TestClient(app):
        pass

    assert calls == [{"json_format": True}]
