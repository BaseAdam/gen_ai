import json
import os
import re

from google import genai
from pydantic import BaseModel, Field

from offers.embeddings import embed_texts
from offers.models import RagAnswer


class EvalCase(BaseModel):
    id: str
    question: str
    relevant: list[str] = Field(description="Titles of the offers that should be returned")
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


class AnswerRecord(BaseModel):
    case_id: str
    question: str
    answer: RagAnswer


def save_answers(records: list[AnswerRecord], path: str) -> None:
    raw_records = [record.model_dump(mode="json") for record in records]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw_records, f, ensure_ascii=False, indent=2)


def load_answers(path: str) -> list[AnswerRecord]:
    with open(path, encoding="utf-8") as f:
        raw_records = json.load(f)
    return [AnswerRecord.model_validate(raw) for raw in raw_records]


CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def cited_numbers(text: str) -> set[int]:
    numbers: set[int] = set()
    for group in CITATION_PATTERN.findall(text):
        numbers.update(int(part) for part in group.split(","))
    return numbers


def invalid_citations(answer: RagAnswer) -> set[int]:
    numbers = cited_numbers(answer.answer)
    return {number for number in numbers if not 1 <= number <= len(answer.sources)}


def cited_titles(answer: RagAnswer) -> set[str]:
    valid = cited_numbers(answer.answer) - invalid_citations(answer)
    return {answer.sources[number - 1].offer.title for number in valid}
