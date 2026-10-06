import time

from google.genai import types

from offers.config import Settings
from offers.io import load_offers, save_offers
from offers.llm import create_client

OFFERS_PATH = "offers.json"

INSTRUCTION = (
    "Napisz realistyczne ogłoszenie o pracę po polsku na podstawie danych w JSON. "
    "Użyj dokładnie czterech sekcji w tej kolejności: O firmie, Obowiązki, Wymagania, Oferujemy. "
    "Każda sekcja zaczyna się od nagłówka w osobnej linii i ma 2-4 zdania zwykłym tekstem. "
    "Sekcje oddziel pustą linią. Nie używaj list ani markdownu. "
    "Nie zmieniaj faktów z JSON-a: firmy, miasta, technologii, widełek i trybu pracy."
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
