import asyncio

import httpx


async def fetch(client: httpx.AsyncClient, url: str) -> int:
    response = await client.get(url)
    return response.status_code


async def main() -> None:
    urls = [
        "https://httpbin.org/delay/2",
        "https://httpbin.org/delay/2",
        "https://httpbin.org/delay/2",
    ]
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[fetch(client, url) for url in urls])
        print(results)


if __name__ == "__main__":
    asyncio.run(main())
