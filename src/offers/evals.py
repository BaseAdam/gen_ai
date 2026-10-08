import json
import os

from google import genai
from pydantic import BaseModel, Field

from offers.embeddings import embed_texts


class EvalCase(BaseModel):
    id: str
    question: str
    relevant: list[str] = Field(description="Tytuły ofert, które powinny wrócić")
    city: str | None = None
    remote: bool | None = None


def load_cases(path: str) -> list[EvalCase]:
    with open(path, encoding="utf-8") as f:
        raw_cases = json.load(f)
    return [EvalCase.model_validate(raw) for raw in raw_cases]


def hits_at_k(retrieved: list[str], relevant: list[str], k: int) -> int:
    return len(set(retrieved[:k]) & set(relevant))


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    shown = retrieved[:k]
    if not shown:
        return 0.0
    return hits_at_k(retrieved, relevant, k) / len(shown)


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    return hits_at_k(retrieved, relevant, k) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    for rank, title in enumerate(retrieved, start=1):
        if title in relevant:
            return 1 / rank
    return 0.0


def embed_questions_cached(
    client: genai.Client, questions: list[str], model: str, path: str
) -> dict[str, list[float]]:
    cache: dict[str, list[float]] = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cache = json.load(f)

    missing = [question for question in questions if question not in cache]
    if missing:
        vectors = embed_texts(client, missing, model, "RETRIEVAL_QUERY")
        cache.update(zip(missing, vectors, strict=True))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)

    return {question: cache[question] for question in questions}
