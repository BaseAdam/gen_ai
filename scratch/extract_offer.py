from offers.config import Settings
from offers.llm import convert_offer_text_to_structure, create_client

EXAMPLE_OFFER = """Mid Python Developer (AI/LLM) – BrightLeaf Analytics
Location: Wroclaw / hybrid work (2 days in the office)
Salary: 16,000 – 21,000 PLN net + VAT (B2B)
We are looking for someone to develop our LLM-based recommendation system.
Requirements: Python, FastAPI, PostgreSQL, experience with the OpenAI API or Gemini, Docker.
Nice to have: LangChain, Kubernetes"""


if __name__ == "__main__":
    settings = Settings()
    client = create_client(settings)
    extracted_offer = convert_offer_text_to_structure(
        client=client, raw_text=EXAMPLE_OFFER, model=settings.gemini_model
    )

    print(extracted_offer.model_dump_json(indent=2))
