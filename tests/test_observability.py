import json
import logging
import time

import pytest
from google.genai import types

from offers.observability import (
    JsonFormatter,
    RequestContext,
    RequestIdFilter,
    configure_logging,
    current_request,
    log_usage,
    record_tokens,
    request_context,
    timed,
)


def fake_clock(monkeypatch: pytest.MonkeyPatch, *ticks: float) -> None:
    values = iter(ticks)
    monkeypatch.setattr(time, "perf_counter", lambda: next(values))


def test_timed_logs_step_duration(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_clock(monkeypatch, 10.0, 10.25)

    with caplog.at_level(logging.INFO, logger="offers"):
        with timed("rag.embed"):
            pass

    assert caplog.messages == ["rag.embed took 250 ms"]
    assert caplog.records[0].step == "rag.embed"
    assert caplog.records[0].duration_ms == 250


def test_timed_logs_failure_and_reraises(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_clock(monkeypatch, 10.0, 11.5)

    with caplog.at_level(logging.INFO, logger="offers"), pytest.raises(ValueError):
        with timed("rag.generate"):
            raise ValueError("boom")

    assert caplog.messages == ["rag.generate failed after 1500 ms: ValueError('boom')"]
    assert caplog.records[0].levelno == logging.ERROR


def test_configure_logging_adds_handler_only_once() -> None:
    configure_logging()
    configure_logging()

    assert len(logging.getLogger("offers").handlers) == 1


def test_log_usage_logs_token_counts(caplog: pytest.LogCaptureFixture) -> None:
    usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=1200, candidates_token_count=80, thoughts_token_count=40
    )
    response = types.GenerateContentResponse(usage_metadata=usage)

    with caplog.at_level(logging.INFO, logger="offers"):
        log_usage("rag.generate", response)

    assert caplog.messages == ["rag.generate tokens: input=1200 output=80 thinking=40"]
    record = caplog.records[0]
    assert (record.input_tokens, record.output_tokens, record.thinking_tokens) == (1200, 80, 40)


def test_log_usage_treats_missing_counts_as_zero(caplog: pytest.LogCaptureFixture) -> None:
    usage = types.GenerateContentResponseUsageMetadata(prompt_token_count=300)
    response = types.GenerateContentResponse(usage_metadata=usage)

    with caplog.at_level(logging.INFO, logger="offers"):
        log_usage("rerank", response)

    assert caplog.messages == ["rerank tokens: input=300 output=0 thinking=0"]


def test_log_usage_skips_response_without_usage(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="offers"):
        log_usage("rerank", types.GenerateContentResponse())

    assert caplog.messages == []


def make_record() -> logging.LogRecord:
    return logging.LogRecord("offers.rag", logging.INFO, __file__, 1, "message", None, None)


def test_request_id_filter_adds_id_of_current_request() -> None:
    record = make_record()
    token = current_request.set(RequestContext(request_id="abc123"))
    try:
        RequestIdFilter().filter(record)
    finally:
        current_request.reset(token)

    assert record.request_id == "abc123"


def test_request_id_filter_marks_logs_outside_requests() -> None:
    record = make_record()

    RequestIdFilter().filter(record)

    assert record.request_id == "-"


def test_configured_handler_fills_request_id() -> None:
    configure_logging()
    [handler] = logging.getLogger("offers").handlers
    record = make_record()

    handler.filter(record)

    assert handler.format(record).endswith(" INFO [-] offers.rag: message")


def test_log_usage_adds_tokens_to_current_request() -> None:
    usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=500, candidates_token_count=100, thoughts_token_count=10
    )
    response = types.GenerateContentResponse(usage_metadata=usage)
    context = RequestContext(request_id="abc123")
    token = current_request.set(context)
    try:
        log_usage("rerank", response)
        log_usage("rag.generate", response)
    finally:
        current_request.reset(token)

    assert (context.input_tokens, context.output_tokens, context.thinking_tokens) == (1000, 200, 20)
    assert context.llm_calls == 2


def test_record_tokens_counts_several_calls_at_once(caplog: pytest.LogCaptureFixture) -> None:
    with request_context("task") as context, caplog.at_level(logging.INFO, logger="offers"):
        record_tokens("pyai.run", 3000, 120, 60, calls=3)

    assert caplog.messages == ["pyai.run tokens: input=3000 output=120 thinking=60"]
    assert caplog.records[0].llm_calls == 3
    assert (context.input_tokens, context.output_tokens, context.thinking_tokens) == (3000, 120, 60)
    assert context.llm_calls == 3


def test_request_context_is_current_only_inside_the_block() -> None:
    with request_context("abc123") as context:
        assert current_request.get() is context
        assert context.request_id == "abc123"

    assert current_request.get() is None


def test_request_context_is_cleared_after_an_error() -> None:
    with pytest.raises(ValueError), request_context("abc123"):
        raise ValueError("boom")

    assert current_request.get() is None


def test_json_formatter_puts_extra_fields_next_to_message() -> None:
    record = logging.makeLogRecord(
        {
            "name": "offers.observability",
            "levelname": "INFO",
            "msg": "%s took %.0f ms",
            "args": ("rag.embed", 312.4),
            "step": "rag.embed",
            "duration_ms": 312,
            "request_id": "abc123",
        }
    )

    data = json.loads(JsonFormatter().format(record))

    assert data["level"] == "INFO"
    assert data["logger"] == "offers.observability"
    assert data["request_id"] == "abc123"
    assert data["message"] == "rag.embed took 312 ms"
    assert data["step"] == "rag.embed"
    assert data["duration_ms"] == 312
    assert "args" not in data
    assert "msecs" not in data


def test_configure_logging_can_switch_to_json(monkeypatch: pytest.MonkeyPatch) -> None:
    app_logger = logging.getLogger("offers")
    monkeypatch.setattr(app_logger, "handlers", [])

    configure_logging(json_format=True)

    [handler] = app_logger.handlers
    assert isinstance(handler.formatter, JsonFormatter)
