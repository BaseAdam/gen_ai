from offers.config import Settings
from offers.io import load_offers
from offers.llm import create_client, stream_interview_tips

if __name__ == "__main__":
    settings = Settings()
    client = create_client(settings)
    offer = load_offers("offers.json")[0]

    for piece in stream_interview_tips(client, offer, settings.gemini_model):
        print(piece, end="", flush=True)
    print()
