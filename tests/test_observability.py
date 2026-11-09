import logging
import time

import pytest
from google.genai import types

from offers.observability import (
    RequestContext,
    RequestIdFilter,
    configure_logging,
    current_request,
    log_usage,
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
