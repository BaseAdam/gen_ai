import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from google.genai import types

LOG_FORMAT = "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"

logger = logging.getLogger(__name__)


@dataclass
class RequestContext:
    request_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0


current_request: ContextVar[RequestContext | None] = ContextVar("current_request", default=None)


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        context = current_request.get()
        record.request_id = context.request_id if context else "-"
        return True


def configure_logging(level: int = logging.INFO) -> None:
    app_logger = logging.getLogger("offers")
    if app_logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.addFilter(RequestIdFilter())
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
    input_tokens = usage.prompt_token_count or 0
    output_tokens = usage.candidates_token_count or 0
    thinking_tokens = usage.thoughts_token_count or 0
    logger.info(
        "%s tokens: input=%d output=%d thinking=%d",
        step,
        input_tokens,
        output_tokens,
        thinking_tokens,
    )
    context = current_request.get()
    if context:
        context.input_tokens += input_tokens
        context.output_tokens += output_tokens
        context.thinking_tokens += thinking_tokens
