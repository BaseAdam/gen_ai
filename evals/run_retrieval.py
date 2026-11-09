import sys
import time

from google import genai

from offers.config import Settings
from offers.evals import (
    EvalCase,
    embed_questions_cached,
    find_regressions,
    load_cases,
    load_thresholds,
    recall_at_k,
    refusal,
    summarize,
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
HOLDOUT = "--holdout" in sys.argv
CASES_PATH = "evals/holdout.json" if HOLDOUT else "evals/retrieval.json"
THRESHOLDS_PATH = "evals/retrieval-thresholds.json"
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


def retrieved_titles(rankings: dict[str, list[ScoredOffer]]) -> dict[str, list[str]]:
    return {name: [item.offer.title for item in ranking] for name, ranking in rankings.items()}


def check_thresholds(averages: dict[str, dict[str, float]]) -> int:
    thresholds = load_thresholds(THRESHOLDS_PATH)
    skipped = [variant for variant in thresholds if variant not in averages]
    failures = find_regressions(averages, thresholds)
    print()
    if skipped:
        print(f"Not checked (variant not run): {', '.join(skipped)}")
    if failures:
        print("REGRESSION:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(f"All thresholds met ({THRESHOLDS_PATH})")
    return 0


def header(label: str) -> str:
    return f"{label:<20}" + "".join(f"{name:>9}" for name in VARIANTS)


def row(label: str, values: list[float]) -> str:
    return f"{label:<20}" + "".join(f"{value:>9.2f}" for value in values)


if __name__ == "__main__":
    settings = Settings()
    client = create_client(settings)
    cases = load_cases(CASES_PATH)
    positives = [case for case in cases if case.relevant]
    negatives = [case for case in cases if not case.relevant]
    vectors = embed_questions_cached(
        client,
        [case.question for case in cases],
        settings.gemini_embedding_model,
        f"evals/vectors-{settings.gemini_embedding_model}.json",
    )

    results: dict[str, list[tuple[EvalCase, list[str]]]] = {name: [] for name in VARIANTS}

    print(f"Cases: {CASES_PATH}")
    if WITH_RERANK:
        print(f"Reranker: {settings.gemini_model}")
    with connect(settings.database_url) as conn:
        print(header("recall@5"))
        for case in positives:
            titles = retrieved_titles(
                run_variants(conn, client, settings, case, vectors[case.question])
            )
            for name in VARIANTS:
                results[name].append((case, titles[name]))
            print(row(case.id, [recall_at_k(titles[name], case.relevant, K) for name in VARIANTS]))
            if WITH_RERANK:
                time.sleep(PAUSE_SECONDS)

        print()
        print(header("refused"))
        for case in negatives:
            titles = retrieved_titles(
                run_variants(conn, client, settings, case, vectors[case.question])
            )
            for name in VARIANTS:
                results[name].append((case, titles[name]))
            print(row(case.id, [refusal(titles[name]) for name in VARIANTS]))
            if WITH_RERANK:
                time.sleep(PAUSE_SECONDS)

    averages = {name: summarize(results[name], K) for name in VARIANTS}
    print()
    print(header("average"))
    for metric in averages[VARIANTS[0]]:
        print(row(metric, [averages[name][metric] for name in VARIANTS]))

    if not HOLDOUT:
        sys.exit(check_thresholds(averages))
