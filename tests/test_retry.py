import asyncio

import httpx
import pytest
from google.genai import errors

from offers.retry import with_retry


class Flaky:
    def __init__(self, failures: list[int | Exception]) -> None:
        self.failures = failures
        self.calls = 0

    async def __call__(self) -> str:
        self.calls += 1
        if self.failures:
            failure = self.failures.pop(0)
            if isinstance(failure, int):
                raise api_error(failure)
            raise failure
        return "ok"


def api_error(code: int) -> errors.APIError:
    return errors.APIError(code, {"error": {"code": code, "message": "test", "status": "TEST"}})


def test_retries_then_succeeds() -> None:
    fn = Flaky([429, 503])

    result = asyncio.run(with_retry(fn, attempts=3, base_delay=0))

    assert result == "ok"
    assert fn.calls == 3


def test_does_not_retry_client_error() -> None:
    fn = Flaky([400])

    with pytest.raises(errors.APIError):
        asyncio.run(with_retry(fn, attempts=3, base_delay=0))

    assert fn.calls == 1


def test_gives_up_after_attempts() -> None:
    fn = Flaky([429, 429, 429, 429])

    with pytest.raises(errors.APIError):
        asyncio.run(with_retry(fn, attempts=3, base_delay=0))

    assert fn.calls == 3


def test_zero_attempts_is_rejected() -> None:
    fn = Flaky([])

    with pytest.raises(ValueError, match="at least 1"):
        asyncio.run(with_retry(fn, attempts=0, base_delay=0))

    assert fn.calls == 0


def test_retries_timeout_then_succeeds() -> None:
    fn = Flaky([httpx.ReadTimeout("too slow")])

    result = asyncio.run(with_retry(fn, attempts=3, base_delay=0))

    assert result == "ok"
    assert fn.calls == 2


def test_does_not_retry_other_errors() -> None:
    fn = Flaky([RuntimeError("bug")])

    with pytest.raises(RuntimeError):
        asyncio.run(with_retry(fn, attempts=3, base_delay=0))

    assert fn.calls == 1
