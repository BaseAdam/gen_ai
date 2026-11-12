import json
import os
import re
from statistics import fmean

from google import genai
from pydantic import BaseModel, Field, TypeAdapter

from offers.embeddings import embed_texts
from offers.io import load_models, save_models
from offers.models import RagAnswer, Verdict


class EvalCase(BaseModel):
    id: str
    question: str
    relevant: list[str] = Field(description="Titles of the offers that should be returned")
    city: str | None = None
    remote: bool | None = None


def load_cases(path: str) -> list[EvalCase]:
    return load_models(path, EvalCase)


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


def refusal(retrieved: list[str]) -> float:
    return 0.0 if retrieved else 1.0


def summarize(results: list[tuple[EvalCase, list[str]]], k: int) -> dict[str, float]:
    positives = [(case, retrieved) for case, retrieved in results if case.relevant]
    negatives = [retrieved for case, retrieved in results if not case.relevant]
    scores = {
        f"recall@{k}": fmean(recall_at_k(r, case.relevant, k) for case, r in positives),
        f"precision@{k}": fmean(precision_at_k(r, case.relevant, k) for case, r in positives),
        "MRR": fmean(reciprocal_rank(r, case.relevant) for case, r in positives),
    }
    if negatives:
        scores["refused"] = fmean(refusal(retrieved) for retrieved in negatives)
    return scores


THRESHOLDS = TypeAdapter(dict[str, dict[str, float]])


def load_thresholds(path: str) -> dict[str, dict[str, float]]:
    with open(path, encoding="utf-8") as f:
        return THRESHOLDS.validate_json(f.read())


def find_regressions(
    averages: dict[str, dict[str, float]], thresholds: dict[str, dict[str, float]]
) -> list[str]:
    failures = []
    for variant, minimums in thresholds.items():
        if variant not in averages:
            continue
        for metric, minimum in minimums.items():
            value = averages[variant][metric]
            if value < minimum:
                failures.append(f"{variant} {metric}: {value:.3f} < {minimum:.2f}")
    return failures


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
    save_models(records, path)


def load_answers(path: str) -> list[AnswerRecord]:
    return load_models(path, AnswerRecord)


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


def relevant_in_context(answer: RagAnswer, relevant: list[str]) -> set[str]:
    return {source.offer.title for source in answer.sources} & set(relevant)


def citation_precision(answer: RagAnswer, relevant: list[str]) -> float | None:
    cited = cited_titles(answer)
    if not cited:
        return None
    return len(cited & set(relevant)) / len(cited)


def citation_recall(answer: RagAnswer, relevant: list[str]) -> float | None:
    expected = relevant_in_context(answer, relevant)
    if not expected:
        return None
    return len(cited_titles(answer) & expected) / len(expected)


class VerdictRecord(BaseModel):
    case_id: str
    verdict: Verdict


def save_verdicts(records: list[VerdictRecord], path: str) -> None:
    save_models(records, path)


def load_verdicts(path: str) -> list[VerdictRecord]:
    return load_models(path, VerdictRecord)
