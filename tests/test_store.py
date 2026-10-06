import pytest

from offers.embeddings import EMBEDDING_DIMENSIONS
from offers.models import Offer
from offers.store import Connection, replace_offers, search_offers


def vec(*head: float) -> list[float]:
    return list(head) + [0.0] * (EMBEDDING_DIMENSIONS - len(head))


@pytest.fixture
def offers() -> list[Offer]:
    return [
        Offer(title="A", company="X", city="Warszawa"),
        Offer(title="B", company="Y", city="Kraków", remote=True, skills=["python"]),
        Offer(title="C", company="Z", city="Gdańsk"),
    ]


CONTENTS = ["text a", "text b", "text c"]
VECTORS = [vec(0.0, 1.0), vec(1.0, 0.0), vec(1.0, 1.0)]


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
