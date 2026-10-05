import os
from collections.abc import Iterator

import psycopg
import pytest

from offers.embeddings import EMBEDDING_DIMENSIONS
from offers.models import Offer
from offers.store import Connection, connect, create_schema, replace_offers, search_offers

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://offers:offers@localhost:5432/offers")
TEST_SCHEMA = "test_store"


def vec(*head: float) -> list[float]:
    return list(head) + [0.0] * (EMBEDDING_DIMENSIONS - len(head))


@pytest.fixture
def conn() -> Iterator[Connection]:
    try:
        connection = connect(DATABASE_URL)
    except psycopg.OperationalError:
        pytest.skip("Postgres is not running")

    connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
    connection.execute(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE")
    connection.execute(f"CREATE SCHEMA {TEST_SCHEMA}")
    connection.execute(f"SET search_path TO {TEST_SCHEMA}, public")
    create_schema(connection)

    yield connection

    connection.rollback()
    connection.execute(f"DROP SCHEMA {TEST_SCHEMA} CASCADE")
    connection.commit()
    connection.close()


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
