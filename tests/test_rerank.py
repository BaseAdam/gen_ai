import json
from typing import Any

from google.genai import types

from offers.models import Offer, Relevance, ScoredOffer
from offers.rerank import rerank


def item(title: str) -> ScoredOffer:
    offer = Offer(title=title, company="X", city="Warsaw")
    return ScoredOffer(offer=offer, score=0.0, chunk=f"fragment {title}")


CANDIDATES = [item("A"), item("B"), item("C")]


def relevance_json(*scores: int) -> str:
    items = [{"index": number, "score": score} for number, score in enumerate(scores, start=1)]
    return json.dumps({"scores": items})


class FakeRerankModels:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> types.GenerateContentResponse:
        self.calls.append(kwargs)
        content = types.Content(role="model", parts=[types.Part(text=self.text)])
        return types.GenerateContentResponse(candidates=[types.Candidate(content=content)])


class FakeRerankClient:
    def __init__(self, text: str) -> None:
        self.models = FakeRerankModels(text)


def test_rerank_sorts_by_relevance_and_drops_irrelevant() -> None:
    client = FakeRerankClient(relevance_json(2, 9, 6))

    ranked = rerank(client, "Where do they provide equipment?", CANDIDATES, "test-model")

    assert [(result.offer.title, result.score) for result in ranked] == [("B", 9), ("C", 6)]


def test_rerank_limits_to_top_k() -> None:
    client = FakeRerankClient(relevance_json(8, 9, 7))

    ranked = rerank(client, "Where do they provide equipment?", CANDIDATES, "test-model", top_k=2)

    assert [result.offer.title for result in ranked] == ["B", "A"]


def test_rerank_sends_numbered_fragments_and_schema() -> None:
    client = FakeRerankClient(relevance_json(5, 5, 5))

    rerank(client, "Where do they provide equipment?", CANDIDATES, "test-model")

    call = client.models.calls[0]
    assert call["model"] == "test-model"
    assert call["contents"].startswith("QUESTION: Where do they provide equipment?")
    assert "[2] fragment B" in call["contents"]
    assert call["config"].response_schema is Relevance


def test_rerank_keeps_original_order_when_response_is_invalid() -> None:
    client = FakeRerankClient("nie wiem")

    ranked = rerank(client, "Where do they provide equipment?", CANDIDATES, "test-model", top_k=2)

    assert [result.offer.title for result in ranked] == ["A", "B"]


def test_rerank_without_candidates_skips_model() -> None:
    client = FakeRerankClient(relevance_json())

    assert rerank(client, "Where do they provide equipment?", [], "test-model") == []
    assert client.models.calls == []
