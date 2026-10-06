from typing import Any

from google.genai import types

from offers.embeddings import EMBEDDING_DIMENSIONS
from offers.models import Offer, ScoredOffer
from offers.rag import RAG_INSTRUCTION, answer_with_rag, build_prompt
from offers.store import Connection, replace_offers


def vec(*head: float) -> list[float]:
    return list(head) + [0.0] * (EMBEDDING_DIMENSIONS - len(head))


class FakeRagModels:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.embed_calls: list[dict[str, Any]] = []
        self.generate_calls: list[dict[str, Any]] = []

    def embed_content(self, **kwargs: Any) -> types.EmbedContentResponse:
        self.embed_calls.append(kwargs)
        embedding = types.ContentEmbedding(values=vec(1.0, 0.0))
        return types.EmbedContentResponse(embeddings=[embedding])

    def generate_content(self, **kwargs: Any) -> types.GenerateContentResponse:
        self.generate_calls.append(kwargs)
        content = types.Content(role="model", parts=[types.Part(text=self.answer)])
        return types.GenerateContentResponse(candidates=[types.Candidate(content=content)])


class FakeRagClient:
    def __init__(self, answer: str) -> None:
        self.models = FakeRagModels(answer)


def test_build_prompt_numbers_offers_and_ends_with_question() -> None:
    results = [
        ScoredOffer(offer=Offer(title="A", company="X", city="Warszawa"), score=0.9),
        ScoredOffer(offer=Offer(title="B", company="Y", city="Kraków"), score=0.8),
    ]

    lines = build_prompt("Co wybrać?", results).splitlines()

    assert lines[0] == "OFERTY:"
    assert lines[1].startswith('[1] {"title":"A"')
    assert lines[2].startswith('[2] {"title":"B"')
    assert lines[-1] == "PYTANIE: Co wybrać?"


def test_answer_with_rag_sends_retrieved_offers_to_model(conn: Connection) -> None:
    offers = [
        Offer(title="A", company="X", city="Warszawa"),
        Offer(title="B", company="Y", city="Kraków"),
        Offer(title="C", company="Z", city="Gdańsk"),
    ]
    replace_offers(conn, offers, ["a", "b", "c"], [vec(0.0, 1.0), vec(1.0, 0.0), vec(1.0, 1.0)])
    client = FakeRagClient("Polecam [1].")

    result = answer_with_rag(client, conn, "Co wybrać?", "chat-model", "embed-model", top_k=2)

    assert result.answer == "Polecam [1]."
    assert [item.offer.title for item in result.sources] == ["B", "C"]

    embed_call = client.models.embed_calls[0]
    assert embed_call["model"] == "embed-model"
    assert embed_call["contents"] == ["Co wybrać?"]
    assert embed_call["config"].task_type == "RETRIEVAL_QUERY"

    generate_call = client.models.generate_calls[0]
    assert generate_call["model"] == "chat-model"
    assert generate_call["contents"] == build_prompt("Co wybrać?", result.sources)
    assert '"title":"A"' not in generate_call["contents"]
    assert generate_call["config"].system_instruction == RAG_INSTRUCTION
