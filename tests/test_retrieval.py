import pytest

from offers.embeddings import EMBEDDING_DIMENSIONS
from offers.models import Offer, ScoredOffer
from offers.retrieval import hybrid_search, reciprocal_rank_fusion
from offers.store import Connection, replace_offers


def vec(*head: float) -> list[float]:
    return list(head) + [0.0] * (EMBEDDING_DIMENSIONS - len(head))


def item(offer_id: int, title: str, chunk: str = "") -> ScoredOffer:
    offer = Offer(title=title, company="X", city="Warsaw")
    return ScoredOffer(offer_id=offer_id, offer=offer, score=0.0, chunk=chunk)


def test_rrf_prefers_offer_found_by_both_rankings() -> None:
    fused = reciprocal_rank_fusion([[item(1, "A"), item(2, "B")], [item(2, "B"), item(3, "C")]])

    assert [result.offer.title for result in fused] == ["B", "A", "C"]


def test_rrf_score_depends_only_on_rank() -> None:
    [result] = reciprocal_rank_fusion([[item(1, "A")]])

    assert result.score == pytest.approx(1 / 61)


def test_rrf_keeps_chunk_from_first_ranking() -> None:
    fused = reciprocal_rank_fusion([[item(1, "A", "semantic")], [item(1, "A", "keyword")]])

    assert fused[0].chunk == "semantic"


def save_offers(conn: Connection) -> None:
    offers = [
        Offer(title="A", company="X", city="Warsaw"),
        Offer(title="B", company="Y", city="Krakow"),
        Offer(title="C", company="Z", city="Gdansk"),
    ]
    contents = [["equipment"], ["coffee"], ["equipment equipment"]]
    vectors = [[vec(0.0, 1.0)], [vec(1.0, 0.0)], [vec(1.0, 1.0)]]
    replace_offers(conn, offers, contents, vectors)


def test_hybrid_search_combines_both_rankings(conn: Connection) -> None:
    save_offers(conn)

    ranked = hybrid_search(conn, vec(1.0, 0.0), "equipment")

    assert [result.offer.title for result in ranked] == ["C", "A", "B"]


def test_hybrid_search_returns_nothing_when_semantic_search_rejects_question(
    conn: Connection,
) -> None:
    save_offers(conn)

    ranked = hybrid_search(conn, vec(0.0, 0.0, 1.0), "equipment", min_score=0.5)

    assert ranked == []
