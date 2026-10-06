import os
from collections.abc import Iterator

import psycopg
import pytest

from offers.store import Connection, connect, create_schema

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://offers:offers@localhost:5432/offers")
TEST_SCHEMA = "test_store"


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
