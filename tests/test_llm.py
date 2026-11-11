import asyncio
import json
from collections.abc import Iterator
from typing import Any

import pytest
from google.genai import types
from pydantic import SecretStr, ValidationError

from offers.config import Settings
from offers.llm import (
    SYSTEM_INSTRUCTION,
    build_http_options,
    convert_offer_text_to_structure,
    stream_interview_tips,
    summarize_all,
    summarize_offer,
)
from offers.models import Offer, OfferSummary


def make_response(
    text: str, input_tokens: int | None = None, output_tokens: int | None = None
) -> types.GenerateContentResponse:

    if (input_tokens is None) or (output_tokens is None):
        metadata = None
    else:
        metadata = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=input_tokens, candidates_token_count=output_tokens
        )

    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(content=types.Content(parts=[types.Part(text=text)], role="model"))
        ],
        usage_metadata=metadata,
    )


class FakeModels:
    def __init__(self, response: types.GenerateContentResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> types.GenerateContentResponse:
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response: types.GenerateContentResponse) -> None:
        self.models = FakeModels(response)


class FakeAsyncModels:
    def __init__(self) -> None:
        self.active = 0
        self.peak = 0

    async def generate_content(self, **kwargs: Any) -> types.GenerateContentResponse:
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        offer = Offer.model_validate_json(kwargs["contents"])
        return make_response(offer.title, 10, 5)


class FakeAsyncClient:
    def __init__(self) -> None:
        self.models = FakeAsyncModels()


@pytest.fixture
def offer() -> Offer:
    return Offer(
        title="AI Engineer",
        company="Lumen AI",
        city="Warsaw",
        salary_min=15000,
        salary_max=22000,
        skills=["python", "openai"],
    )


def test_summary_maps_text_and_tokens(offer: Offer) -> None:
    client = FakeClient(make_response("Short summary.", 80, 20))

    summary = summarize_offer(client, offer, "test-model")

    assert summary == OfferSummary(text="Short summary.", input_tokens=80, output_tokens=20)


def test_summary_without_metadata(offer: Offer) -> None:
    client = FakeClient(make_response("Test answer"))

    summary = summarize_offer(client, offer, "test-model")

    assert summary == OfferSummary(text="Test answer", input_tokens=0, output_tokens=0)


def test_empty_response(offer: Offer) -> None:
    client = FakeClient(types.GenerateContentResponse())

    summary = summarize_offer(client, offer, "test-model")

    assert summary == OfferSummary(text="No summary", input_tokens=0, output_tokens=0)


def test_summarize_all_respects_limit_and_order() -> None:
    offers = [Offer(title=f"Offer {i}", company="Acme", city="Warsaw") for i in range(10)]
    client = FakeAsyncClient()

    summaries = asyncio.run(summarize_all(client, offers, "test-model", max_concurrency=3))

    assert client.models.peak == 3
    assert [s.text for s in summaries] == [o.title for o in offers]


def test_sends_offer_json_and_instruction(offer: Offer) -> None:
    client = FakeClient(make_response("Answer"))

    summarize_offer(client, offer, "test-model")

    assert len(client.models.calls) == 1
    call = client.models.calls[0]
    assert call["model"] == "test-model"
    assert call["contents"] == offer.model_dump_json()
    assert call["config"].system_instruction == SYSTEM_INSTRUCTION


def test_extraction_returns_validated_offer() -> None:
    model_output = json.dumps(
        {
            "title": "Mid Python Developer",
            "company": "BrightLeaf",
            "city": "Wroclaw",
            "salary_min": 16000,
            "salary_max": 21000,
            "remote": False,
            "skills": ["Python", "FastAPI"],
            "skills_nice_to_have": ["LangChain"],
        }
    )
    client = FakeClient(make_response(model_output))

    offer = convert_offer_text_to_structure(client, "raw job ad", "test-model")

    assert offer.salary_min == 16000
    assert offer.skills == ["python", "fastapi"]
    assert offer.skills_nice_to_have == ["langchain"]


def test_salary_range_inverted() -> None:
    model_output = json.dumps(
        {
            "title": "Mid Python Developer",
            "company": "BrightLeaf",
            "city": "Wroclaw",
            "salary_min": 36000,
            "salary_max": 21000,
            "remote": False,
            "skills": ["Python", "FastAPI"],
            "skills_nice_to_have": ["LangChain"],
        }
    )

    client = FakeClient(make_response(model_output))
    with pytest.raises(ValidationError):
        convert_offer_text_to_structure(client, "raw job ad", "test-model")


def test_empty_response_on_extraction() -> None:
    client = FakeClient(types.GenerateContentResponse())

    with pytest.raises(ValueError, match="Empty response"):
        convert_offer_text_to_structure(client, "", "test-model")


def test_extraction_sends_raw_text_and_schema() -> None:
    client = FakeClient(make_response('{"title": "Dev", "company": "X", "city": "Y"}'))

    convert_offer_text_to_structure(client, "Answer", "test-model")

    assert len(client.models.calls) == 1
    call = client.models.calls[0]
    assert call["model"] == "test-model"
    assert call["contents"] == "Answer"
    assert call["config"].response_schema is Offer
    assert call["config"].response_mime_type == "application/json"


class FakeStreamModels:
    def __init__(self, chunks: list[types.GenerateContentResponse]) -> None:
        self.chunks = chunks

    def generate_content_stream(self, **kwargs: Any) -> Iterator[types.GenerateContentResponse]:
        return iter(self.chunks)


class FakeStreamClient:
    def __init__(self, chunks: list[types.GenerateContentResponse]) -> None:
        self.models = FakeStreamModels(chunks)


def test_stream_yields_text_pieces_and_skips_empty(offer: Offer) -> None:
    chunks = [make_response("1. Review "), types.GenerateContentResponse(), make_response("SQL.")]
    client = FakeStreamClient(chunks)

    pieces = list(stream_interview_tips(client, offer, "test-model"))

    assert pieces == ["1. Review ", "SQL."]


def test_client_retries_overloaded_model_but_not_slow_requests() -> None:
    settings = Settings(
        gemini_api_key=SecretStr("test-key"),
        gemini_model="test-model",
        gemini_timeout_seconds=30,
        gemini_retry_attempts=4,
        _env_file=None,
    )

    options = build_http_options(settings)

    assert options.timeout == 30000
    assert options.retry_options is not None
    assert options.retry_options.attempts == 4
    assert 503 in options.retry_options.http_status_codes
    assert 504 not in options.retry_options.http_status_codes
