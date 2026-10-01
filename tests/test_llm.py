import asyncio
from typing import Any

import pytest
from google.genai import types

from offers.llm import SYSTEM_INSTRUCTION, summarize_all, summarize_offer
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
        city="Warszawa",
        salary_min=15000,
        salary_max=22000,
        skills=["python", "openai"],
    )


def test_summary_maps_text_and_tokens(offer: Offer) -> None:
    client = FakeClient(make_response("Krótkie podsumowanie.", 80, 20))

    summary = summarize_offer(client, offer, "test-model")

    assert summary == OfferSummary(text="Krótkie podsumowanie.", input_tokens=80, output_tokens=20)


def test_summary_without_metadata(offer: Offer) -> None:
    client = FakeClient(make_response("Testowa odpowiedz"))

    summary = summarize_offer(client, offer, "test-model")

    assert summary == OfferSummary(text="Testowa odpowiedz", input_tokens=0, output_tokens=0)


def test_empty_response(offer: Offer) -> None:
    client = FakeClient(types.GenerateContentResponse())

    summary = summarize_offer(client, offer, "test-model")

    assert summary == OfferSummary(text="Brak podsumowania", input_tokens=0, output_tokens=0)


def test_summarize_all_respects_limit_and_order() -> None:
    offers = [Offer(title=f"Oferta {i}", company="Acme", city="Warszawa") for i in range(10)]
    client = FakeAsyncClient()

    summaries = asyncio.run(summarize_all(client, offers, "test-model", max_concurrency=3))

    assert client.models.peak == 3
    assert [s.text for s in summaries] == [o.title for o in offers]


def test_sends_offer_json_and_instruction(offer: Offer) -> None:
    client = FakeClient(make_response("Odpowiedź"))

    summarize_offer(client, offer, "test-model")

    assert len(client.models.calls) == 1
    call = client.models.calls[0]
    assert call["model"] == "test-model"
    assert call["contents"] == offer.model_dump_json()
    assert call["config"].system_instruction == SYSTEM_INSTRUCTION
