import asyncio
from collections.abc import Iterator

from google import genai
from google.genai import types
from google.genai.client import AsyncClient
from google.genai.types import GenerateContentConfig, GenerateContentResponse

from offers.config import Settings
from offers.models import Offer, OfferSummary

SYSTEM_INSTRUCTION = (
    "role: it recruiter, task: summarize offer, format: 2 sentences, language: english, "
    "content: (company, salary range, important skills)"
)

EXTRACTION_INSTRUCTION = (
    "use data which is inside the input text, if there are no information provided "
    "about some property then fill with Null, salary ranges should be as integers monthly in PLN, "
    "remote offer should be true ONLY if offer text are saying it"
)


RETRY_STATUS_CODES = [429, 500, 502, 503]


def build_http_options(settings: Settings) -> types.HttpOptions:
    retry = types.HttpRetryOptions(
        attempts=settings.gemini_retry_attempts,
        initial_delay=1.0,
        max_delay=10.0,
        http_status_codes=RETRY_STATUS_CODES,
    )
    return types.HttpOptions(
        timeout=int(settings.gemini_timeout_seconds * 1000), retry_options=retry
    )


def create_client(settings: Settings) -> genai.Client:
    return genai.Client(
        api_key=settings.gemini_api_key.get_secret_value(),
        http_options=build_http_options(settings),
    )


def _convert_response_to_summary(response: GenerateContentResponse) -> OfferSummary:
    usage = response.usage_metadata
    if usage is None:
        input_tokens, output_tokens = 0, 0
    else:
        input_tokens = usage.prompt_token_count or 0
        output_tokens = usage.candidates_token_count or 0

    return OfferSummary(
        text=response.text or "No summary",
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


def _generate_structured_content_config() -> GenerateContentConfig:
    config = types.GenerateContentConfig(
        system_instruction=EXTRACTION_INSTRUCTION,
        temperature=0,
        response_mime_type="application/json",
        response_schema=Offer,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    return config


def convert_offer_text_to_structure(client: genai.Client, raw_text: str, model: str) -> Offer:
    response = client.models.generate_content(
        model=model, contents=raw_text, config=_generate_structured_content_config()
    )
    text = response.text
    if text is None:
        raise ValueError("Empty response")
    return Offer.model_validate_json(text)


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


TIPS_INSTRUCTION = (
    "Based on the job offer, list 5 short tips on how to prepare "
    "for a job interview for this position. Answer in English."
)


def stream_interview_tips(client: genai.Client, offer: Offer, model: str) -> Iterator[str]:
    config = types.GenerateContentConfig(
        system_instruction=TIPS_INSTRUCTION,
        temperature=0.3,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    stream = client.models.generate_content_stream(
        model=model, contents=offer.model_dump_json(), config=config
    )
    for chunk in stream:
        if chunk.text:
            yield chunk.text
