import psycopg
from psycopg.rows import TupleRow

from offers.embeddings import EMBEDDING_DIMENSIONS
from offers.models import Offer, ScoredOffer

type Connection = psycopg.Connection[TupleRow]

CREATE_TABLE = f"""
CREATE TABLE IF NOT EXISTS offers (
    id bigserial PRIMARY KEY,
    data jsonb NOT NULL,
    content text NOT NULL,
    embedding vector({EMBEDDING_DIMENSIONS}) NOT NULL
)
"""

INSERT_OFFER = """
INSERT INTO offers (data, content, embedding)
VALUES (%s::jsonb, %s, %s::vector)
"""

SEARCH_OFFERS = """
SELECT data, 1 - (embedding <=> %(query)s::vector) AS score
FROM offers
WHERE (%(city)s::text IS NULL OR lower(data->>'city') = %(city)s::text)
  AND (%(remote)s::boolean IS NULL OR (data->>'remote')::boolean = %(remote)s::boolean)
  AND (
    %(min_score)s::float8 IS NULL
    OR 1 - (embedding <=> %(query)s::vector) >= %(min_score)s::float8
  )
ORDER BY embedding <=> %(query)s::vector
LIMIT %(top_k)s
"""


def connect(database_url: str) -> Connection:
    return psycopg.connect(database_url)


def create_schema(conn: Connection) -> None:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute(CREATE_TABLE)
    conn.commit()


def replace_offers(
    conn: Connection, offers: list[Offer], contents: list[str], vectors: list[list[float]]
) -> None:
    rows = [
        (offer.model_dump_json(), content, str(vector))
        for offer, content, vector in zip(offers, contents, vectors, strict=True)
    ]
    conn.execute("TRUNCATE offers RESTART IDENTITY")
    conn.cursor().executemany(INSERT_OFFER, rows)
    conn.commit()


def search_offers(
    conn: Connection,
    query_vector: list[float],
    top_k: int = 5,
    city: str | None = None,
    remote: bool | None = None,
    min_score: float | None = None,
) -> list[ScoredOffer]:
    params = {
        "query": str(query_vector),
        "top_k": top_k,
        "city": city.lower() if city else None,
        "remote": remote,
        "min_score": min_score,
    }
    rows = conn.execute(SEARCH_OFFERS, params).fetchall()
    return [ScoredOffer(offer=Offer.model_validate(data), score=score) for data, score in rows]
