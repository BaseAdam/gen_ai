import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import psycopg
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from google.genai import errors, types
from psycopg_pool import PoolTimeout
from pydantic import SecretStr
from pydantic_ai import ModelHTTPError, ModelTimeoutError
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

import offers.api
from offers.api import (
    app,
    get_agent_model,
    get_client,
    get_conn,
    get_offers,
    get_settings,
    mount_frontend,
)
from offers.config import Settings
from offers.models import Offer
from offers.observability import current_request
from offers.pyai_agent import ANSWER_TOOL
from offers.store import Connection, replace_offers
from tests.helpers import call_response, text_response, vec

OFFERS = [
    Offer(title="A", company="X", city="Warsaw", salary_min=10000, salary_max=20000),
    Offer(title="B", company="Y", city="Krakow", remote=True, skills=["python"]),
]

CHUNKS = [["a"], ["b"]]


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


def test_model_deadline_returns_504(api: TestClient, fake_client: FakeClient) -> None:
    body = {"error": {"code": 504, "message": "Deadline expired", "status": "DEADLINE_EXCEEDED"}}
    fake_client.models.error = errors.APIError(504, body)

    response = api.post("/ask", json={"question": "How much do they pay in Warsaw?"})

    assert response.status_code == 504
    assert response.json() == {"detail": "The model did not respond in time"}


def test_unreachable_model_returns_502(api: TestClient, fake_client: FakeClient) -> None:
    fake_client.models.error = httpx.ConnectError("connection refused")

    response = api.post("/ask", json={"question": "How much do they pay in Warsaw?"})

    assert response.status_code == 502
    assert response.json() == {"detail": "Model service error"}


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


class FakePool:
    instances: list[FakePool] = []

    @staticmethod
    def check_connection(conn: object) -> None:
        pass

    def __init__(self, conninfo: str, **options: Any) -> None:
        self.conninfo = conninfo
        self.options = options
        self.events: list[str] = []
        FakePool.instances.append(self)

    def __enter__(self) -> FakePool:
        self.events.append("open")
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.events.append("close")

    @contextmanager
    def connection(self) -> Iterator[str]:
        self.events.append("borrow")
        try:
            yield "pooled connection"
        finally:
            self.events.append("return")


def test_startup_configures_logging_and_opens_the_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        gemini_api_key=SecretStr("test-key"),
        gemini_model="test-model",
        log_json=True,
        database_url="postgresql://user:pass@db:5432/offers",
        _env_file=None,
    )
    calls: list[dict[str, Any]] = []
    FakePool.instances.clear()
    monkeypatch.setattr(offers.api, "get_settings", lambda: settings)
    monkeypatch.setattr(offers.api, "configure_logging", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(offers.api, "ConnectionPool", FakePool)

    with TestClient(app):
        [pool] = FakePool.instances
        assert pool.events == ["open"]

    assert calls == [{"json_format": True}]
    assert pool.conninfo == "postgresql://user:pass@db:5432/offers"
    assert pool.options["max_size"] == offers.api.POOL_MAX_SIZE
    assert pool.options["check"] is FakePool.check_connection
    assert pool.events == ["open", "close"]


def test_each_request_borrows_a_connection_and_gives_it_back() -> None:
    pool = FakePool("conninfo")
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(pool=pool)))

    dependency = get_conn(request)  # type: ignore[arg-type]
    assert next(dependency) == "pooled connection"
    assert pool.events == ["borrow"]
    with pytest.raises(StopIteration):
        next(dependency)

    assert pool.events == ["borrow", "return"]


@pytest.mark.parametrize("path", ["/rag", "/agent"])
def test_returns_503_when_no_pooled_connection_is_free(api: TestClient, path: str) -> None:
    def busy_pool() -> Connection:
        raise PoolTimeout("couldn't get a connection after 5.00 sec")

    app.dependency_overrides[get_conn] = busy_pool

    response = api.post(path, json={"question": "Best pick?"})

    assert response.status_code == 503


USAGE = RequestUsage(input_tokens=1000, output_tokens=50, output_reasoning_tokens=30)


def tool_call(tool: str, **args: Any) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool, args)], usage=USAGE)


def final_answer(answer: str, offers: list[str]) -> ModelResponse:
    return tool_call(ANSWER_TOOL, answer=answer, offers=offers)


type Respond = Callable[[list[ModelMessage], AgentInfo], ModelResponse]


def use_agent(respond: Respond) -> None:
    app.dependency_overrides[get_agent_model] = lambda: FunctionModel(respond)
    app.dependency_overrides[get_conn] = lambda: "unused connection"


def script(*responses: ModelResponse) -> Respond:
    queue = list(responses)
    return lambda messages, info: queue.pop(0)


def test_agent_returns_answer_offers_steps_and_usage(api: TestClient) -> None:
    use_agent(
        script(tool_call("list_offers", city="Krakow"), final_answer("B is in Krakow.", ["B"]))
    )

    response = api.post("/agent", json={"question": "What is there in Krakow?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "B is in Krakow.",
        "offers": [OFFERS[1].model_dump(mode="json")],
        "steps": [["list_offers"], []],
        "usage": {"llm_calls": 2, "input_tokens": 2000, "output_tokens": 40, "thinking_tokens": 60},
    }


def test_agent_returns_each_real_offer_once(
    api: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    use_agent(script(final_answer("B, B again and C.", ["B", "C", "B"])))

    with caplog.at_level(logging.WARNING, logger="offers"):
        response = api.post("/agent", json={"question": "Which offers?"})

    assert [offer["title"] for offer in response.json()["offers"]] == ["B"]
    assert "The agent named offers that do not exist: ['C']" in caplog.messages


def test_agent_rejects_short_question_without_calling_model(api: TestClient) -> None:
    calls: list[list[ModelMessage]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(messages)
        return final_answer("Hi", [])

    use_agent(respond)

    response = api.post("/agent", json={"question": "?"})

    assert response.status_code == 422
    assert calls == []


def test_agent_that_cannot_finish_returns_502(api: TestClient) -> None:
    use_agent(lambda messages, info: tool_call("list_offers"))

    response = api.post("/agent", json={"question": "Loop forever"})

    assert response.status_code == 502
    assert response.json() == {"detail": "The agent could not finish the answer"}


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (ModelHTTPError(504, "test-model"), 504),
        (ModelTimeoutError("test-model", "read timed out"), 504),
        (ModelHTTPError(429, "test-model", body={"message": "internal detail"}), 502),
    ],
)
def test_agent_model_errors_map_to_gateway_statuses(
    api: TestClient, error: Exception, status: int
) -> None:
    def fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise error

    use_agent(fail)

    response = api.post("/agent", json={"question": "How much does B pay?"})

    assert response.status_code == status
    assert "internal detail" not in response.text


def app_with_frontend(directory: Path) -> TestClient:
    frontend_app = FastAPI()

    @frontend_app.get("/offers")
    def offers() -> list[str]:
        return ["A"]

    mount_frontend(frontend_app, directory)
    return TestClient(frontend_app)


def test_serves_the_built_frontend_next_to_the_api(tmp_path: Path) -> None:
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text('<div id="root"></div>')
    (tmp_path / "assets" / "app.js").write_text("console.log('hi')")
    client = app_with_frontend(tmp_path)

    page = client.get("/")
    script = client.get("/assets/app.js")

    assert page.status_code == 200
    assert page.text == '<div id="root"></div>'
    assert page.headers["content-type"].startswith("text/html")
    assert script.text == "console.log('hi')"
    assert client.get("/offers").json() == ["A"]
    assert client.get("/missing.js").status_code == 404


def test_without_a_frontend_build_only_the_api_is_served(tmp_path: Path) -> None:
    client = app_with_frontend(tmp_path / "dist")

    assert client.get("/").status_code == 404
    assert client.get("/offers").json() == ["A"]
