from typing import Any

import pytest
from google.genai import types

from offers.embeddings import cosine_similarity, embed_texts, offer_to_text, rank_offers
from offers.models import Offer


def test_offer_to_text_describes_offer() -> None:
    offer = Offer(title="Dev", company="Acme", city="Gdańsk", remote=True, skills=["Python", "SQL"])

    assert offer_to_text(offer) == "Dev w Acme, Gdańsk, zdalnie. Wymagania: python, sql."


def test_offer_to_text_without_skills() -> None:
    offer = Offer(title="Dev", company="Acme", city="Gdańsk")

    assert offer_to_text(offer) == "Dev w Acme, Gdańsk, stacjonarnie. Wymagania: brak danych."


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([1.0, 0.0], [1.0, 0.0], 1.0),
        ([1.0, 0.0], [0.0, 1.0], 0.0),
        ([1.0, 0.0], [-1.0, 0.0], -1.0),
        ([3.0, 4.0], [6.0, 8.0], 1.0),
    ],
)
def test_cosine_similarity(a: list[float], b: list[float], expected: float) -> None:
    assert cosine_similarity(a, b) == pytest.approx(expected)


def test_cosine_similarity_rejects_different_lengths() -> None:
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 2.0], [1.0])


class FakeEmbedModels:
    def __init__(self, vectors: list[list[float] | None]) -> None:
        self.vectors = vectors
        self.calls: list[dict[str, Any]] = []

    def embed_content(self, **kwargs: Any) -> types.EmbedContentResponse:
        self.calls.append(kwargs)
        embeddings = [types.ContentEmbedding(values=vector) for vector in self.vectors]
        return types.EmbedContentResponse(embeddings=embeddings)


class FakeEmbedClient:
    def __init__(self, vectors: list[list[float] | None]) -> None:
        self.models = FakeEmbedModels(vectors)


def test_embed_texts_returns_vectors_and_sends_config() -> None:
    client = FakeEmbedClient([[1.0, 0.0], [0.0, 1.0]])

    vectors = embed_texts(client, ["a", "b"], "test-model", "RETRIEVAL_DOCUMENT")

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    call = client.models.calls[0]
    assert call["model"] == "test-model"
    assert call["contents"] == ["a", "b"]
    assert call["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert call["config"].output_dimensionality == 768


@pytest.mark.parametrize("vectors", [[[1.0, 0.0]], [[1.0, 0.0], None]])
def test_embed_texts_rejects_incomplete_response(vectors: list[list[float] | None]) -> None:
    client = FakeEmbedClient(vectors)

    with pytest.raises(ValueError, match="Incomplete"):
        embed_texts(client, ["a", "b"], "test-model", "RETRIEVAL_DOCUMENT")


@pytest.fixture
def offers() -> list[Offer]:
    return [
        Offer(title="A", company="X", city="Warszawa"),
        Offer(title="B", company="Y", city="Kraków"),
        Offer(title="C", company="Z", city="Gdańsk"),
    ]


VECTORS = [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]


def test_rank_offers_sorts_by_similarity(offers: list[Offer]) -> None:
    ranked = rank_offers([1.0, 0.0], VECTORS, offers)

    assert [item.offer.title for item in ranked] == ["B", "C", "A"]
    assert ranked[0].score == pytest.approx(1.0)
    assert ranked[2].score == pytest.approx(0.0)


def test_rank_offers_limits_to_top_k(offers: list[Offer]) -> None:
    ranked = rank_offers([1.0, 0.0], VECTORS, offers, top_k=2)

    assert [item.offer.title for item in ranked] == ["B", "C"]


def test_rank_offers_rejects_mismatched_lengths(offers: list[Offer]) -> None:
    with pytest.raises(ValueError):
        rank_offers([1.0, 0.0], VECTORS[:2], offers)
