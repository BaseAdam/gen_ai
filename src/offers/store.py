import psycopg
from psycopg.rows import TupleRow

from offers.embeddings import EMBEDDING_DIMENSIONS
from offers.models import Offer, ScoredOffer

type Connection = psycopg.Connection[TupleRow]
type Vector = list[float]

CREATE_OFFERS = """
CREATE TABLE IF NOT EXISTS offers (
    id bigserial PRIMARY KEY,
    data jsonb NOT NULL
)
"""

CREATE_CHUNKS = f"""
CREATE TABLE IF NOT EXISTS chunks (
    id bigserial PRIMARY KEY,
    offer_id bigint NOT NULL REFERENCES offers (id) ON DELETE CASCADE,
    content text NOT NULL,
    embedding vector({EMBEDDING_DIMENSIONS}) NOT NULL
)
"""

INSERT_OFFER = "INSERT INTO offers (data) VALUES (%s::jsonb) RETURNING id"

INSERT_CHUNK = "INSERT INTO chunks (offer_id, content, embedding) VALUES (%s, %s, %s::vector)"


SEARCH_OFFERS = """
SELECT data, content, score
FROM (
    SELECT DISTINCT ON (offers.id)
        offers.data,
        chunks.content,
        1 - (chunks.embedding <=> %(query)s::vector) AS score
    FROM chunks
    JOIN offers ON offers.id = chunks.offer_id
    WHERE (%(city)s::text IS NULL OR lower(offers.data->>'city') = %(city)s::text)
      AND (%(remote)s::boolean IS NULL OR (offers.data->>'remote')::boolean = %(remote)s::boolean)
    ORDER BY offers.id, chunks.embedding <=> %(query)s::vector
) AS best
WHERE %(min_score)s::float8 IS NULL OR score >= %(min_score)s::float8
ORDER BY score DESC
LIMIT %(top_k)s
"""


def connect(database_url: str) -> Connection:
    return psycopg.connect(database_url)


def create_schema(conn: Connection) -> None:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute(CREATE_OFFERS)
    conn.execute(CREATE_CHUNKS)
    conn.commit()


def reset_schema(conn: Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS chunks, offers")
    create_schema(conn)


def replace_offers(
    conn: Connection, offers: list[Offer], chunks: list[list[str]], vectors: list[list[Vector]]
) -> None:
    conn.execute("TRUNCATE offers, chunks RESTART IDENTITY")
    for offer, contents, embeddings in zip(offers, chunks, vectors, strict=True):
        [row] = conn.execute(INSERT_OFFER, (offer.model_dump_json(),)).fetchall()
        offer_id = row[0]
        rows = [
            (offer_id, content, str(embedding))
            for content, embedding in zip(contents, embeddings, strict=True)
        ]
        conn.cursor().executemany(INSERT_CHUNK, rows)
    conn.commit()


def search_offers(
    conn: Connection,
    query_vector: Vector,
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
    return [
        ScoredOffer(offer=Offer.model_validate(data), score=score, chunk=content)
        for data, content, score in rows
    ]
