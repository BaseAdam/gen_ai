import sys
import time
from statistics import fmean

from google import genai

from offers.config import Settings
from offers.evals import (
    EvalCase,
    embed_questions_cached,
    load_cases,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from offers.llm import create_client
from offers.models import ScoredOffer
from offers.rag import RERANK_CANDIDATES
from offers.rerank import rerank
from offers.retrieval import hybrid_search
from offers.store import Connection, connect, search_keywords, search_offers

K = 5
PAUSE_SECONDS = 5
WITH_RERANK = "--rerank" in sys.argv
VARIANTS = ["vectors", "bm25", "hybrid"] + (["rerank"] if WITH_RERANK else [])


def run_variants(
    conn: Connection,
    client: genai.Client,
    settings: Settings,
    case: EvalCase,
    vector: list[float],
) -> dict[str, list[ScoredOffer]]:
    min_score = settings.rag_min_score
    candidates = hybrid_search(
        conn,
        vector,
        case.question,
        RERANK_CANDIDATES,
        city=case.city,
        remote=case.remote,
        min_score=min_score,
    )
    rankings = {
        "vectors": search_offers(
            conn, vector, K, city=case.city, remote=case.remote, min_score=min_score
        ),
        "bm25": search_keywords(conn, case.question, K, city=case.city, remote=case.remote),
        "hybrid": candidates[:K],
    }
    if WITH_RERANK:
        rankings["rerank"] = rerank(client, case.question, candidates, settings.gemini_model, K)
    return rankings


def header(label: str) -> str:
    return f"{label:<20}" + "".join(f"{name:>9}" for name in VARIANTS)


def row(label: str, values: list[float]) -> str:
    return f"{label:<20}" + "".join(f"{value:>9.2f}" for value in values)


if __name__ == "__main__":
    settings = Settings()
    client = create_client(settings)
    cases = load_cases("evals/retrieval.json")
    positives = [case for case in cases if case.relevant]
    negatives = [case for case in cases if not case.relevant]
    vectors = embed_questions_cached(
        client,
        [case.question for case in cases],
        settings.gemini_embedding_model,
        f"evals/vectors-{settings.gemini_embedding_model}.json",
    )

    recall: dict[str, list[float]] = {name: [] for name in VARIANTS}
    precision: dict[str, list[float]] = {name: [] for name in VARIANTS}
    rr: dict[str, list[float]] = {name: [] for name in VARIANTS}
    refused: dict[str, list[float]] = {name: [] for name in VARIANTS}

    if WITH_RERANK:
        print(f"Reranker: {settings.gemini_model}")
    with connect(settings.database_url) as conn:
        print(header("recall@5"))
        for case in positives:
            rankings = run_variants(conn, client, settings, case, vectors[case.question])
            for name, ranking in rankings.items():
                retrieved = [item.offer.title for item in ranking]
                recall[name].append(recall_at_k(retrieved, case.relevant, K))
                precision[name].append(precision_at_k(retrieved, case.relevant, K))
                rr[name].append(reciprocal_rank(retrieved, case.relevant))
            print(row(case.id, [recall[name][-1] for name in VARIANTS]))
            if WITH_RERANK:
                time.sleep(PAUSE_SECONDS)

        print()
        print(header("refused"))
        for case in negatives:
            rankings = run_variants(conn, client, settings, case, vectors[case.question])
            for name, ranking in rankings.items():
                refused[name].append(0.0 if ranking else 1.0)
            print(row(case.id, [refused[name][-1] for name in VARIANTS]))
            if WITH_RERANK:
                time.sleep(PAUSE_SECONDS)

    summary = [
        ("recall@5", recall),
        ("precision@5", precision),
        ("MRR", rr),
        ("refused", refused),
    ]
    print()
    print(header("average"))
    for label, scores in summary:
        print(row(label, [fmean(scores[name]) for name in VARIANTS]))
