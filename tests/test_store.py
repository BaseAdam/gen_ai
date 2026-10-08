import pytest

from offers.embeddings import EMBEDDING_DIMENSIONS
from offers.models import Offer
from offers.store import Connection, replace_offers, search_keywords, search_offers


def vec(*head: float) -> list[float]:
    return list(head) + [0.0] * (EMBEDDING_DIMENSIONS - len(head))


@pytest.fixture
def offers() -> list[Offer]:
    return [
        Offer(title="A", company="X", city="Warszawa"),
        Offer(title="B", company="Y", city="Kraków", remote=True, skills=["python"]),
        Offer(title="C", company="Z", city="Gdańsk"),
    ]


CONTENTS = [["text a"], ["text b"], ["text c"]]
VECTORS = [[vec(0.0, 1.0)], [vec(1.0, 0.0)], [vec(1.0, 1.0)]]

KEYWORD_CONTENTS = [["kawa i biuro"], ["sprzęt oraz biuro oraz kawa", "sprzęt sprzęt"], ["sprzęt"]]
KEYWORD_VECTORS = [[vec(1.0)], [vec(1.0), vec(1.0)], [vec(1.0)]]


def test_search_returns_closest_first(conn: Connection, offers: list[Offer]) -> None:
    replace_offers(conn, offers, CONTENTS, VECTORS)

    ranked = search_offers(conn, vec(1.0, 0.0))

    assert [item.offer.title for item in ranked] == ["B", "C", "A"]
    assert [item.score for item in ranked] == pytest.approx([1.0, 0.7071, 0.0], abs=1e-4)


def test_search_returns_saved_offer_unchanged(conn: Connection, offers: list[Offer]) -> None:
    replace_offers(conn, offers, CONTENTS, VECTORS)

    [best] = search_offers(conn, vec(1.0, 0.0), top_k=1)

    assert best.offer == offers[1]


def test_replace_offers_overwrites_previous_rows(conn: Connection, offers: list[Offer]) -> None:
    replace_offers(conn, offers, CONTENTS, VECTORS)
    replace_offers(conn, offers[:1], CONTENTS[:1], VECTORS[:1])

    ranked = search_offers(conn, vec(1.0, 0.0))

    assert [item.offer.title for item in ranked] == ["A"]


def test_replace_offers_rejects_mismatched_lengths(conn: Connection, offers: list[Offer]) -> None:
    with pytest.raises(ValueError):
        replace_offers(conn, offers, CONTENTS, VECTORS[:2])


def test_search_filters_by_city_ignoring_case(conn: Connection, offers: list[Offer]) -> None:
    replace_offers(conn, offers, CONTENTS, VECTORS)

    ranked = search_offers(conn, vec(1.0, 0.0), city="WARSZAWA")

    assert [item.offer.title for item in ranked] == ["A"]


def test_search_filters_by_remote(conn: Connection, offers: list[Offer]) -> None:
    replace_offers(conn, offers, CONTENTS, VECTORS)

    remote = search_offers(conn, vec(1.0, 0.0), remote=True)
    onsite = search_offers(conn, vec(1.0, 0.0), remote=False)

    assert [item.offer.title for item in remote] == ["B"]
    assert [item.offer.title for item in onsite] == ["C", "A"]


def test_search_filters_before_limit(conn: Connection, offers: list[Offer]) -> None:
    replace_offers(conn, offers, CONTENTS, VECTORS)

    ranked = search_offers(conn, vec(1.0, 0.0), top_k=1, city="Gdańsk")

    assert [item.offer.title for item in ranked] == ["C"]


def test_search_drops_results_below_min_score(conn: Connection, offers: list[Offer]) -> None:
    replace_offers(conn, offers, CONTENTS, VECTORS)

    ranked = search_offers(conn, vec(1.0, 0.0), min_score=0.5)

    assert [item.offer.title for item in ranked] == ["B", "C"]


def test_search_returns_nothing_for_unrelated_query(conn: Connection, offers: list[Offer]) -> None:
    replace_offers(conn, offers, CONTENTS, VECTORS)

    ranked = search_offers(conn, vec(0.0, 0.0, 1.0), min_score=0.5)

    assert ranked == []


def test_search_returns_each_offer_once_with_its_best_chunk(
    conn: Connection, offers: list[Offer]
) -> None:
    contents = [["a1", "a2"], ["b1"], ["c1"]]
    vectors = [[vec(0.0, 1.0), vec(1.0, 0.0)], [vec(1.0, 1.0)], [vec(0.0, 1.0)]]
    replace_offers(conn, offers, contents, vectors)

    ranked = search_offers(conn, vec(1.0, 0.0))

    assert [(item.offer.title, item.chunk) for item in ranked] == [
        ("A", "a2"),
        ("B", "b1"),
        ("C", "c1"),
    ]


def test_search_keywords_ranks_offers_by_best_chunk(conn: Connection, offers: list[Offer]) -> None:
    replace_offers(conn, offers, KEYWORD_CONTENTS, KEYWORD_VECTORS)

    ranked = search_keywords(conn, "sprzęt")

    assert [(item.offer.title, item.chunk) for item in ranked] == [
        ("B", "sprzęt sprzęt"),
        ("C", "sprzęt"),
    ]


def test_search_keywords_applies_filters(conn: Connection, offers: list[Offer]) -> None:
    replace_offers(conn, offers, KEYWORD_CONTENTS, KEYWORD_VECTORS)

    ranked = search_keywords(conn, "sprzęt", city="Kraków")

    assert [item.offer.title for item in ranked] == ["B"]


def test_both_searches_return_the_same_offer_id(conn: Connection, offers: list[Offer]) -> None:
    replace_offers(conn, offers, [["kawa"], ["sprzęt"], ["biuro"]], VECTORS)

    [semantic] = search_offers(conn, vec(1.0, 0.0), top_k=1)
    [keyword] = search_keywords(conn, "sprzęt")

    assert semantic.offer.title == keyword.offer.title == "B"
    assert semantic.offer_id == keyword.offer_id != 0


def test_create_schema_adds_vector_index(conn: Connection) -> None:
    rows = conn.execute(
        "SELECT indexdef FROM pg_indexes "
        "WHERE schemaname = current_schema() AND tablename = 'chunks'"
    ).fetchall()

    assert any("hnsw" in row[0] and "vector_cosine_ops" in row[0] for row in rows)


def test_search_widens_hnsw_candidate_list(conn: Connection, offers: list[Offer]) -> None:
    replace_offers(conn, offers, CONTENTS, VECTORS)

    search_offers(conn, vec(1.0, 0.0))

    [row] = conn.execute("SHOW hnsw.ef_search").fetchall()
    assert row[0] == "100"
