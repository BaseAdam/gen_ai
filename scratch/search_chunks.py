from offers.config import Settings
from offers.embeddings import embed_texts
from offers.llm import create_client
from offers.store import connect, search_offers

QUESTION = "Where will I design database queries?"

if __name__ == "__main__":
    settings = Settings()
    client = create_client(settings)

    [vector] = embed_texts(client, [QUESTION], settings.gemini_embedding_model, "RETRIEVAL_QUERY")

    with connect(settings.database_url) as conn:
        for item in search_offers(conn, vector, top_k=3):
            print(f"{item.score:.3f}  {item.offer.title} @ {item.offer.company}")
            print(item.chunk)
            print()
