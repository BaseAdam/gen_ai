import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

from google.genai import types

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    app_logger = logging.getLogger("offers")
    if app_logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    app_logger.addHandler(handler)
    app_logger.setLevel(level)


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


@contextmanager
def timed(step: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    except Exception as error:
        logger.error("%s failed after %.0f ms: %r", step, elapsed_ms(start), error)
        raise
    logger.info("%s took %.0f ms", step, elapsed_ms(start))


def log_usage(step: str, response: types.GenerateContentResponse) -> None:
    usage = response.usage_metadata
    if usage is None:
        return
    logger.info(
        "%s tokens: input=%d output=%d thinking=%d",
        step,
        usage.prompt_token_count or 0,
        usage.candidates_token_count or 0,
        usage.thoughts_token_count or 0,
    )
