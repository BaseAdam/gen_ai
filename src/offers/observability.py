import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from google.genai import types
from opentelemetry import trace

LOG_FORMAT = "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"

STANDARD_RECORD_FIELDS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("offers")


@dataclass
class RequestContext:
    request_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    llm_calls: int = 0
    steps: list[list[str]] = field(default_factory=list)

    @property
    def tool_calls(self) -> list[str]:
        return [tool for step in self.steps for tool in step]


current_request: ContextVar[RequestContext | None] = ContextVar("current_request", default=None)


@contextmanager
def request_context(request_id: str) -> Iterator[RequestContext]:
    context = RequestContext(request_id=request_id)
    token = current_request.set(context)
    try:
        yield context
    finally:
        current_request.reset(token)


class RequestIdFilter(logging.Filter):
    """Adds the request ID and, when tracing is on, the trace ID, so logs lead to their trace."""

    def filter(self, record: logging.LogRecord) -> bool:
        context = current_request.get()
        record.request_id = context.request_id if context else "-"
        span = trace.get_current_span().get_span_context()
        record.trace_id = format(span.trace_id, "032x") if span.is_valid else "-"
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, object] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "trace_id": getattr(record, "trace_id", "-"),
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in STANDARD_RECORD_FIELDS and key not in data:
                data[key] = value
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False, default=str)


def configure_logging(level: int = logging.INFO, json_format: bool = False) -> None:
    app_logger = logging.getLogger("offers")
    if app_logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter() if json_format else logging.Formatter(LOG_FORMAT))
    handler.addFilter(RequestIdFilter())
    app_logger.addHandler(handler)
    app_logger.setLevel(level)
    app_logger.propagate = False


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


@contextmanager
def timed(step: str) -> Iterator[None]:
    """Logs how long a step took and traces it as a span (a no-op until tracing is configured)."""
    start = time.perf_counter()
    with tracer.start_as_current_span(step):
        try:
            yield
        except Exception as error:
            duration = elapsed_ms(start)
            logger.error(
                "%s failed after %.0f ms: %r",
                step,
                duration,
                error,
                extra={"step": step, "duration_ms": round(duration), "error": repr(error)},
            )
            raise
        duration = elapsed_ms(start)
        logger.info(
            "%s took %.0f ms",
            step,
            duration,
            extra={"step": step, "duration_ms": round(duration)},
        )


def record_tokens(
    step: str, input_tokens: int, output_tokens: int, thinking_tokens: int, calls: int = 1
) -> None:
    logger.info(
        "%s tokens: input=%d output=%d thinking=%d",
        step,
        input_tokens,
        output_tokens,
        thinking_tokens,
        extra={
            "step": step,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
            "llm_calls": calls,
        },
    )
    # An event rather than attributes: one span can record several model calls.
    trace.get_current_span().add_event(
        "llm.usage",
        {
            "step": step,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
            "llm_calls": calls,
        },
    )
    context = current_request.get()
    if context:
        context.input_tokens += input_tokens
        context.output_tokens += output_tokens
        context.thinking_tokens += thinking_tokens
        context.llm_calls += calls


def log_usage(step: str, response: types.GenerateContentResponse) -> None:
    usage = response.usage_metadata
    if usage is None:
        return
    record_tokens(
        step,
        usage.prompt_token_count or 0,
        usage.candidates_token_count or 0,
        usage.thoughts_token_count or 0,
    )


def record_step(caller: str, calls: list[tuple[str, dict[str, Any]]]) -> None:
    for tool, args in calls:
        logger.info(
            "%s calls %s(%s)",
            caller,
            tool,
            json.dumps(args, ensure_ascii=False, default=str),
            extra={"tool": tool},
        )
    context = current_request.get()
    if context:
        context.steps.append([tool for tool, _ in calls])
