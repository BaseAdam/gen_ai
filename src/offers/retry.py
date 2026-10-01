import asyncio
import random
from collections.abc import Awaitable, Callable

from google.genai import errors

RETRYABLE_CODES = {429, 500, 502, 503, 504}


async def with_retry[T](
    fn: Callable[[], Awaitable[T]], attempts: int = 3, base_delay: float = 1.0
) -> T:
    for attempt in range(attempts):
        try:
            return await fn()
        except errors.APIError as e:
            if e.code not in RETRYABLE_CODES or attempt == attempts - 1:
                raise
            delay = base_delay * 2**attempt + random.uniform(0, base_delay)
            await asyncio.sleep(delay)

    raise ValueError("attempts must be at least 1")
