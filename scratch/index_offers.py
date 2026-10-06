from offers.chunking import chunk_offer
from offers.config import Settings
from offers.embeddings import embed_groups
from offers.io import load_offers
from offers.llm import create_client
from offers.store import connect, replace_offers, reset_schema

if __name__ == "__main__":
    settings = Settings()
    client = create_client(settings)
    offers = load_offers("offers.json")

    chunks = [chunk_offer(offer) for offer in offers]
    vectors = embed_groups(client, chunks, settings.gemini_embedding_model, "RETRIEVAL_DOCUMENT")

    with connect(settings.database_url) as conn:
        reset_schema(conn)
        replace_offers(conn, offers, chunks, vectors)

    print(f"Zapisano {len(offers)} ofert i {sum(len(group) for group in chunks)} fragmentów")
