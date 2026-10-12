from offers.assistant import answer_question
from offers.config import Settings
from offers.io import load_offers
from offers.llm import create_client

QUESTION = "Where are salaries better: in Warsaw or in Krakow?"

if __name__ == "__main__":
    settings = Settings()
    client = create_client(settings)
    offers = load_offers("offers.json")

    print(answer_question(client, QUESTION, offers, settings.gemini_model))
