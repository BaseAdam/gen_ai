import time

from google.genai import types

from offers.config import Settings
from offers.io import load_offers, save_offers
from offers.llm import create_client

OFFERS_PATH = "offers.json"

INSTRUCTION = (
    "Write a realistic job ad in English based on the JSON data. "
    "Use exactly four sections in this order: About the company, Responsibilities, "
    "Requirements, What we offer. Each section starts with its heading on a separate line "
    "and has 2-4 sentences of plain text. Separate sections with a blank line. "
    "Do not use lists or markdown. Do not change the facts from the JSON: "
    "company, city, technologies, salary range and work mode."
)


if __name__ == "__main__":
    settings = Settings()
    client = create_client(settings)
    offers = load_offers(OFFERS_PATH)
    config = types.GenerateContentConfig(
        system_instruction=INSTRUCTION,
        temperature=0.8,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    for number, offer in enumerate(offers, start=1):
        if offer.description:
            continue

        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=offer.model_dump_json(exclude={"description"}),
            config=config,
        )
        offer.description = (response.text or "").strip()
        save_offers(offers, OFFERS_PATH)
        print(f"{number}/{len(offers)} {offer.title}")
        time.sleep(4)
