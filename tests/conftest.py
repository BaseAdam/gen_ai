import logging
import os
from collections.abc import Iterator

import psycopg
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from offers.store import Connection, connect, create_schema

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://offers:offers@localhost:5432/offers")
TEST_SCHEMA = "test_store"

# OpenTelemetry lets a process set the global tracer provider only once, so the tests share one.
SPANS = InMemorySpanExporter()


@pytest.fixture(autouse=True)
def restore_app_logger() -> Iterator[None]:
    app_logger = logging.getLogger("offers")
    handlers, level, propagate = list(app_logger.handlers), app_logger.level, app_logger.propagate
    yield
    app_logger.handlers[:] = handlers
    app_logger.setLevel(level)
    app_logger.propagate = propagate


@pytest.fixture
def spans() -> Iterator[InMemorySpanExporter]:
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(SPANS))
        trace.set_tracer_provider(provider)
    SPANS.clear()
    yield SPANS
    SPANS.clear()


@pytest.fixture
def conn() -> Iterator[Connection]:
    try:
        connection = connect(DATABASE_URL)
    except psycopg.OperationalError:
        if os.environ.get("CI"):
            raise  # in CI a missing database is a failure, not a reason to skip
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
