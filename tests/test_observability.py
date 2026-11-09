import logging
import time

import pytest

from offers.observability import configure_logging, timed


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
