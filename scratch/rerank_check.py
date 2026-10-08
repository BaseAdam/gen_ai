from offers.config import Settings
from offers.embeddings import embed_texts
from offers.llm import create_client
from offers.rerank import rerank
from offers.retrieval import hybrid_search
from offers.store import connect

QUESTIONS = [
    "W których ofertach firma zapewnia sprzęt do pracy zdalnej?",
    "Praca dla kierowcy ciężarówki",
]

if __name__ == "__main__":
    settings = Settings()
    client = create_client(settings)

    vectors = embed_texts(client, QUESTIONS, settings.gemini_embedding_model, "RETRIEVAL_QUERY")

    with connect(settings.database_url) as conn:
        for question, vector in zip(QUESTIONS, vectors, strict=True):
            candidates = hybrid_search(
                conn, vector, question, top_k=10, min_score=settings.rag_min_score
            )
            reranked = rerank(client, question, candidates, settings.gemini_model)

            print(question)
            print("  przed:", ", ".join(item.offer.company for item in candidates[:5]))
            for item in reranked:
                print(f"  {item.score:>2}  {item.offer.title} @ {item.offer.company}")
            if not reranked:
                print("  (brak trafnych ofert)")
            print()
