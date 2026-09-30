from offers.config import Settings
from offers.io import load_offers
from offers.llm import create_client, summarize_offer

if __name__ == "__main__":
    settings = Settings()
    client = create_client(settings)
    offers = load_offers("offers.json")
    summary = summarize_offer(client, offers[0], settings.gemini_model)

    print(summary)
