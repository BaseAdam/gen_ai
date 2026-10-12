from offers.config import Settings
from offers.embeddings import embed_texts
from offers.llm import create_client
from offers.retrieval import hybrid_search
from offers.store import connect, search_keywords, search_offers

QUESTION = "Which offers provide equipment for remote work?"

if __name__ == "__main__":
    settings = Settings()
    client = create_client(settings)

    [vector] = embed_texts(client, [QUESTION], settings.gemini_embedding_model, "RETRIEVAL_QUERY")

    with connect(settings.database_url) as conn:
        rankings = {
            "vectors": search_offers(conn, vector, top_k=5),
            "keywords": search_keywords(conn, QUESTION, top_k=5),
            "hybrid": hybrid_search(conn, vector, QUESTION, top_k=5),
        }

    for name, ranking in rankings.items():
        print(name)
        for position, item in enumerate(ranking, start=1):
            print(f"  {position}. {item.offer.title} @ {item.offer.company}")
