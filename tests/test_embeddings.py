from typing import Any

import pytest
from google.genai import types

from offers.embeddings import embed_groups, embed_texts, offer_to_text
from offers.models import Offer


def test_offer_to_text_describes_offer() -> None:
    offer = Offer(title="Dev", company="Acme", city="Gdansk", remote=True, skills=["Python", "SQL"])

    assert offer_to_text(offer) == "Dev at Acme, Gdansk, remote. Requirements: python, sql."


def test_offer_to_text_without_skills() -> None:
    offer = Offer(title="Dev", company="Acme", city="Gdansk")

    assert offer_to_text(offer) == "Dev at Acme, Gdansk, on-site. Requirements: not specified."


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


def test_embed_groups_uses_one_call_and_keeps_grouping() -> None:
    client = FakeEmbedClient([[1.0], [2.0], [3.0]])

    vectors = embed_groups(client, [["a", "b"], ["c"]], "test-model", "RETRIEVAL_DOCUMENT")

    assert vectors == [[[1.0], [2.0]], [[3.0]]]
    assert len(client.models.calls) == 1
    assert client.models.calls[0]["contents"] == ["a", "b", "c"]
