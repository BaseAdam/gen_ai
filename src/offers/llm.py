import asyncio

from google import genai
from google.genai import types
from google.genai.client import AsyncClient
from google.genai.types import GenerateContentConfig, GenerateContentResponse

from offers.config import Settings
from offers.models import Offer, OfferSummary

SYSTEM_INSTRUCTION = (
    "role: it recruiter, task: summarize offer, format: 2 sentences, language: polish, "
    "content: (company, salary range, important skills)"
)


def create_client(settings: Settings) -> genai.Client:
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    return client


def _convert_response_to_summary(response: GenerateContentResponse) -> OfferSummary:
    usage = response.usage_metadata
    if usage is None:
        input_tokens, output_tokens = 0, 0
    else:
        input_tokens = usage.prompt_token_count or 0
        output_tokens = usage.candidates_token_count or 0

    return OfferSummary(
        text=response.text or "Brak podsumowania",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _generate_content_config() -> GenerateContentConfig:
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.1,
        max_output_tokens=10000,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    return config


def summarize_offer(client: genai.Client, offer: Offer, model: str) -> OfferSummary:
    response = client.models.generate_content(
        model=model, contents=offer.model_dump_json(), config=_generate_content_config()
    )
    return _convert_response_to_summary(response)


async def summarize_offer_async(aclient: AsyncClient, offer: Offer, model: str) -> OfferSummary:
    response = await aclient.models.generate_content(
        model=model, contents=offer.model_dump_json(), config=_generate_content_config()
    )

    return _convert_response_to_summary(response)


async def summarize_all(
    aclient: AsyncClient, offers: list[Offer], model: str, max_concurrency: int = 5
) -> list[OfferSummary]:
    semaphore = asyncio.Semaphore(max_concurrency)

    async def limited(offer: Offer) -> OfferSummary:
        async with semaphore:
            return await summarize_offer_async(aclient, offer, model)

    return await asyncio.gather(*[limited(offer) for offer in offers])
