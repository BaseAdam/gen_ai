from offers.config import Settings
from offers.llm import convert_offer_text_to_structure, create_client

EXAMPLE_OFFER = """Mid Python Developer (AI/LLM) – BrightLeaf Analytics
Lokalizacja: Wrocław / praca hybrydowa (2 dni w biurze)
Widełki: 16 000 – 21 000 PLN netto + VAT (B2B)
Szukamy osoby, która rozwinie nasz system rekomendacji oparty na LLM.
Wymagania: Python, FastAPI, PostgreSQL, doświadczenie z OpenAI API lub Gemini, Docker.
Mile widziane: LangChain, Kubernetes"""


if __name__ == "__main__":
    settings = Settings()
    client = create_client(settings)
    extracted_offer = convert_offer_text_to_structure(
        client=client, raw_text=EXAMPLE_OFFER, model=settings.gemini_model
    )

    print(extracted_offer.model_dump_json(indent=2))
