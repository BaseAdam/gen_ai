from offers.config import Settings
from offers.llm import create_client
from offers.rag import answer_with_rag
from offers.store import connect

QUESTION = "Which offers provide equipment for remote work?"

if __name__ == "__main__":
    settings = Settings()
    client = create_client(settings)

    with connect(settings.database_url) as conn:
        result = answer_with_rag(
            client,
            conn,
            QUESTION,
            settings.gemini_model,
            settings.gemini_embedding_model,
            min_score=settings.rag_min_score,
            use_rerank=settings.rag_rerank,
        )

    print(result.answer)
    print()
    for number, item in enumerate(result.sources, start=1):
        print(f"[{number}] {item.score:>4}  {item.offer}")
