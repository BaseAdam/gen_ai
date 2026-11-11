from offers.models import ScoredOffer
from offers.store import Connection, search_keywords, search_offers

RRF_K = 60
CANDIDATES = 20


def reciprocal_rank_fusion(rankings: list[list[ScoredOffer]], k: int = RRF_K) -> list[ScoredOffer]:
    scores: dict[int, float] = {}
    hits: dict[int, list[tuple[int, ScoredOffer]]] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item.offer_id] = scores.get(item.offer_id, 0.0) + 1 / (k + rank)
            hits.setdefault(item.offer_id, []).append((rank, item))

    fused = []
    for offer_id, score in scores.items():
        found = sorted(hits[offer_id], key=lambda hit: hit[0])
        chunks = list(dict.fromkeys(item.chunk for _, item in found))
        best = found[0][1]
        fused.append(best.model_copy(update={"score": score, "chunk": "\n\n".join(chunks)}))
    return sorted(fused, key=lambda item: item.score, reverse=True)


def hybrid_search(
    conn: Connection,
    query_vector: list[float],
    question: str,
    top_k: int = 5,
    city: str | None = None,
    remote: bool | None = None,
    min_score: float | None = None,
) -> list[ScoredOffer]:
    semantic = search_offers(
        conn, query_vector, CANDIDATES, city=city, remote=remote, min_score=min_score
    )
    if not semantic:
        return []
    keyword = search_keywords(conn, question, CANDIDATES, city=city, remote=remote)
    return reciprocal_rank_fusion([semantic, keyword])[:top_k]
