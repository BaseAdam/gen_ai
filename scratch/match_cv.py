from offers.config import Settings
from offers.embeddings import embed_texts, offer_to_text, rank_offers
from offers.io import load_offers
from offers.llm import create_client

CV = """
Fullstack developer z trzyletnim doświadczeniem w TypeScript i React.
Uczę się Pythona, FastAPI i pracy z LLM. Znam Dockera i podstawy SQL.
Szukam pracy w Warszawie albo zdalnie.
"""


if __name__ == "__main__":
    settings = Settings()
    client = create_client(settings)
    offers = load_offers("offers.json")
    model = settings.gemini_embedding_model

    texts = [offer_to_text(offer) for offer in offers]
    offer_vectors = embed_texts(client, texts, model, "RETRIEVAL_DOCUMENT")
    [cv_vector] = embed_texts(client, [CV], model, "RETRIEVAL_QUERY")

    for item in rank_offers(cv_vector, offer_vectors, offers):
        print(f"{item.score:.3f}  {item.offer}")
