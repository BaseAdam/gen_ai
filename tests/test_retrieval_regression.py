from offers.chunking import chunk_offer
from offers.embeddings import EMBEDDING_DIMENSIONS
from offers.evals import EvalCase, find_regressions, load_cases, load_thresholds, summarize
from offers.io import load_offers
from offers.store import Connection, replace_offers, search_keywords

K = 5
PLACEHOLDER_VECTOR = [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)


def test_keyword_search_meets_thresholds_on_dev_set(conn: Connection) -> None:
    offers = load_offers("offers.json")
    chunks = [chunk_offer(offer) for offer in offers]
    vectors = [[PLACEHOLDER_VECTOR] * len(group) for group in chunks]
    replace_offers(conn, offers, chunks, vectors)

    results: list[tuple[EvalCase, list[str]]] = []
    for case in load_cases("evals/retrieval.json"):
        ranking = search_keywords(conn, case.question, K, city=case.city, remote=case.remote)
        results.append((case, [item.offer.title for item in ranking]))

    averages = {"bm25": summarize(results, K)}
    thresholds = load_thresholds("evals/retrieval-thresholds.json")

    assert find_regressions(averages, {"bm25": thresholds["bm25"]}) == []
