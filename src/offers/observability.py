import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

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
