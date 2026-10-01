import asyncio
import time

from offers.config import Settings
from offers.io import load_offers
from offers.llm import create_client, summarize_all


async def main() -> None:
    settings = Settings()
    client = create_client(settings)
    offers = load_offers("offers.json")[:5]

    async with client.aio as aclient:
        start = time.perf_counter()
        summaries = await summarize_all(aclient, offers, settings.gemini_model)
        elapsed = time.perf_counter() - start

    for offer, summary in zip(offers, summaries, strict=True):
        print(f"{offer.title} @ {offer.company}")
        print(f"  {summary.text}\n")

    total_tokens = sum(s.input_tokens + s.output_tokens for s in summaries)
    print(f"{len(summaries)} ofert w {elapsed:.1f} s, tokeny łącznie: {total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
