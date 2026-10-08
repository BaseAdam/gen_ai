import json
from typing import Any

from google.genai import types

from offers.embeddings import EMBEDDING_DIMENSIONS
from offers.models import Offer, ScoredOffer
from offers.rag import NO_RESULTS_ANSWER, RAG_INSTRUCTION, answer_with_rag, build_prompt
from offers.store import Connection, replace_offers


def vec(*head: float) -> list[float]:
    return list(head) + [0.0] * (EMBEDDING_DIMENSIONS - len(head))


def save_offers(conn: Connection) -> None:
    offers = [
        Offer(title="A", company="X", city="Warszawa"),
        Offer(title="B", company="Y", city="Kraków"),
        Offer(title="C", company="Z", city="Gdańsk"),
    ]
    vectors = [[vec(0.0, 1.0)], [vec(1.0, 0.0)], [vec(1.0, 1.0)]]
    replace_offers(conn, offers, [["a"], ["b"], ["c"]], vectors)


class FakeRagModels:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.relevance = '{"scores": []}'
        self.embed_calls: list[dict[str, Any]] = []
        self.generate_calls: list[dict[str, Any]] = []

    def embed_content(self, **kwargs: Any) -> types.EmbedContentResponse:
        self.embed_calls.append(kwargs)
        embedding = types.ContentEmbedding(values=vec(1.0, 0.0))
        return types.EmbedContentResponse(embeddings=[embedding])

    def generate_content(self, **kwargs: Any) -> types.GenerateContentResponse:
        self.generate_calls.append(kwargs)
        is_rerank = kwargs["config"].response_schema is not None
        text = self.relevance if is_rerank else self.answer
        content = types.Content(role="model", parts=[types.Part(text=text)])
        return types.GenerateContentResponse(candidates=[types.Candidate(content=content)])


class FakeRagClient:
    def __init__(self, answer: str) -> None:
        self.models = FakeRagModels(answer)


def test_build_prompt_numbers_offers_and_ends_with_question() -> None:
    results = [
        ScoredOffer(offer=Offer(title="A", company="X", city="Warszawa"), score=0.9),
        ScoredOffer(offer=Offer(title="B", company="Y", city="Kraków"), score=0.8),
    ]

    prompt = build_prompt("Co wybrać?", results)

    assert prompt.startswith('OFERTY:\n[1] {"title":"A"')
    assert '\n\n[2] {"title":"B"' in prompt
    assert prompt.endswith("\n\nPYTANIE: Co wybrać?")


def test_build_prompt_uses_matched_chunk_instead_of_full_description() -> None:
    offer = Offer(
        title="A",
        company="X",
        city="Warszawa",
        description="O firmie\nDługi opis firmy.\n\nOferujemy\nOwoce w biurze.",
    )
    results = [ScoredOffer(offer=offer, score=0.9, chunk="A w X\nOferujemy\nOwoce w biurze.")]

    prompt = build_prompt("Co dają?", results)

    assert "Fragment ogłoszenia:\nA w X\nOferujemy\nOwoce w biurze." in prompt
    assert "Długi opis firmy." not in prompt
    assert '"description"' not in prompt


def test_answer_with_rag_sends_retrieved_offers_to_model(conn: Connection) -> None:
    save_offers(conn)
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
    assert generate_call["config"].thinking_config.thinking_level == types.ThinkingLevel.LOW


def test_answer_with_rag_applies_filters(conn: Connection) -> None:
    save_offers(conn)
    client = FakeRagClient("Polecam [1].")

    result = answer_with_rag(
        client, conn, "Co wybrać?", "chat-model", "embed-model", city="Gdańsk", remote=False
    )

    assert [item.offer.title for item in result.sources] == ["C"]
    assert '"title":"B"' not in client.models.generate_calls[0]["contents"]


def test_answer_with_rag_skips_model_when_nothing_matches(conn: Connection) -> None:
    save_offers(conn)
    client = FakeRagClient("Polecam [1].")

    result = answer_with_rag(client, conn, "Co wybrać?", "chat-model", "embed-model", city="Poznań")

    assert result.answer == NO_RESULTS_ANSWER
    assert result.sources == []
    assert client.models.generate_calls == []


def test_answer_with_rag_applies_min_score(conn: Connection) -> None:
    save_offers(conn)
    client = FakeRagClient("Polecam [1].")

    result = answer_with_rag(client, conn, "Co wybrać?", "chat-model", "embed-model", min_score=0.8)

    assert [item.offer.title for item in result.sources] == ["B"]


def relevance(*scores: int) -> str:
    items = [{"index": number, "score": score} for number, score in enumerate(scores, start=1)]
    return json.dumps({"scores": items})


def test_answer_with_rag_reranks_candidates(conn: Connection) -> None:
    save_offers(conn)
    client = FakeRagClient("Polecam [1].")
    client.models.relevance = relevance(3, 4, 9)

    result = answer_with_rag(
        client, conn, "Co wybrać?", "chat-model", "embed-model", use_rerank=True
    )

    assert [item.offer.title for item in result.sources] == ["A"]
    assert len(client.models.generate_calls) == 2
    assert '"title":"B"' not in client.models.generate_calls[1]["contents"]


def test_answer_with_rag_skips_model_when_reranker_rejects_everything(conn: Connection) -> None:
    save_offers(conn)
    client = FakeRagClient("Polecam [1].")
    client.models.relevance = relevance(1, 2, 0)

    result = answer_with_rag(
        client, conn, "Co wybrać?", "chat-model", "embed-model", use_rerank=True
    )

    assert result.answer == NO_RESULTS_ANSWER
    assert len(client.models.generate_calls) == 1
