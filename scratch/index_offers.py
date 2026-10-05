from offers.config import Settings
from offers.embeddings import embed_texts, offer_to_text
from offers.io import load_offers
from offers.llm import create_client
from offers.store import connect, create_schema, replace_offers

if __name__ == "__main__":
    settings = Settings()
    client = create_client(settings)
    offers = load_offers("offers.json")

    contents = [offer_to_text(offer) for offer in offers]
    vectors = embed_texts(client, contents, settings.gemini_embedding_model, "RETRIEVAL_DOCUMENT")

    with connect(settings.database_url) as conn:
        create_schema(conn)
        replace_offers(conn, offers, contents, vectors)

    print(f"Zapisano {len(offers)} ofert")
