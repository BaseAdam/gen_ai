from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from google.genai import errors, types
from pydantic import SecretStr

from offers.api import app, get_client, get_offers, get_settings
from offers.config import Settings
from offers.models import Offer

OFFERS = [
    Offer(title="A", company="X", city="Warszawa", salary_min=10000, salary_max=20000),
    Offer(title="B", company="Y", city="Kraków", remote=True, skills=["python"]),
]


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


class FakeClient:
    def __init__(self) -> None:
        self.models = FakeModels()


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def api(fake_client: FakeClient) -> Iterator[TestClient]:
    settings = Settings(
        gemini_api_key=SecretStr("test-key"), gemini_model="test-model", _env_file=None
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
    response = api.get("/offers", params={"city": "kraków", "remote": "true"})

    assert [offer["title"] for offer in response.json()] == ["B"]


def test_extract_returns_offer(api: TestClient, fake_client: FakeClient) -> None:
    raw = '{"title": "Dev", "company": "Acme", "city": "Gdańsk", "skills": ["Python"]}'
    fake_client.models.responses = [text_response(raw)]

    response = api.post("/offers/extract", json={"text": "Szukamy Python deva do Acme w Gdańsku"})

    assert response.status_code == 200
    assert response.json()["company"] == "Acme"
    assert response.json()["skills"] == ["python"]
    assert fake_client.models.calls[0]["model"] == "test-model"


def test_extract_rejects_short_text_without_calling_model(
    api: TestClient, fake_client: FakeClient
) -> None:
    response = api.post("/offers/extract", json={"text": "za krótko"})

    assert response.status_code == 422
    assert fake_client.models.calls == []


def test_ask_runs_tool_and_returns_answer(api: TestClient, fake_client: FakeClient) -> None:
    fake_client.models.responses = [
        call_response("salary_stats", city="Warszawa"),
        text_response("Średnio 15000 PLN"),
    ]

    response = api.post("/ask", json={"question": "Ile płacą w Warszawie?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "Średnio 15000 PLN"}
    assert len(fake_client.models.calls) == 2


def test_tips_streams_text(api: TestClient, fake_client: FakeClient) -> None:
    fake_client.models.chunks = ["Powtórz ", "", "asyncio."]

    response = api.get("/offers/0/tips")

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    assert response.text == "Powtórz asyncio."


@pytest.mark.parametrize("index", [2, -1])
def test_tips_unknown_offer_returns_404(
    api: TestClient, fake_client: FakeClient, index: int
) -> None:
    response = api.get(f"/offers/{index}/tips")

    assert response.status_code == 404
    assert response.json() == {"detail": "Nie ma takiej oferty"}
    assert fake_client.models.calls == []


def test_model_timeout_returns_504(api: TestClient, fake_client: FakeClient) -> None:
    fake_client.models.error = httpx.ReadTimeout("too slow")

    response = api.post("/ask", json={"question": "Ile płacą w Warszawie?"})

    assert response.status_code == 504
    assert response.json() == {"detail": "Model nie odpowiedział na czas"}


def test_model_error_returns_502_without_details(api: TestClient, fake_client: FakeClient) -> None:
    body = {"error": {"code": 503, "message": "internal detail", "status": "UNAVAILABLE"}}
    fake_client.models.error = errors.APIError(503, body)

    response = api.post("/offers/extract", json={"text": "Szukamy Python deva do Acme w Gdańsku"})

    assert response.status_code == 502
    assert "internal detail" not in response.text
