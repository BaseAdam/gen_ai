from google import genai
from google.genai import types

from offers.config import Settings
from offers.models import Offer, OfferSummary

SYSTEM_INSTRUCTION = (
    "role: it recruiter, task: summarize offer, format: 2 sentences, language: polish, "
    "content: (company, salary range, important skills)"
)


def create_client(settings: Settings) -> genai.Client:
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    return client


def summarize_offer(client: genai.Client, offer: Offer, model: str) -> OfferSummary:
    response = client.models.generate_content(
        model=model,
        contents=offer.model_dump_json(),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.1,
            max_output_tokens=10000,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )

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
