import asyncio
import random
from collections.abc import Awaitable, Callable

import httpx
from google.genai import errors

RETRYABLE_CODES = {429, 500, 502, 503, 504}


def is_retryable(error: Exception) -> bool:
    if isinstance(error, errors.APIError):
        return error.code in RETRYABLE_CODES
    return isinstance(error, httpx.TimeoutException)


async def with_retry[T](
    fn: Callable[[], Awaitable[T]], attempts: int = 3, base_delay: float = 1.0
) -> T:
    for attempt in range(attempts):
        try:
            return await fn()
        except (errors.APIError, httpx.TimeoutException) as e:
            if not is_retryable(e) or attempt == attempts - 1:
                raise
            delay = base_delay * 2**attempt + random.uniform(0, base_delay)
            await asyncio.sleep(delay)

    raise ValueError("attempts must be at least 1")
