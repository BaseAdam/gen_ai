from offers.config import Settings
from offers.embeddings import embed_texts
from offers.llm import create_client
from offers.store import connect, search_offers

QUESTIONS = [
    "I'm looking for a job as a Python developer",
    "Offers for a frontend developer with React",
    "Remote machine learning job",
    "A job for a truck driver",
    "A recipe for Ukrainian pierogi",
    "What is the capital of France?",
]

if __name__ == "__main__":
    settings = Settings()
    client = create_client(settings)

    vectors = embed_texts(client, QUESTIONS, settings.gemini_embedding_model, "RETRIEVAL_QUERY")

    with connect(settings.database_url) as conn:
        for question, vector in zip(QUESTIONS, vectors, strict=True):
            results = search_offers(conn, vector, top_k=3)
            scores = "  ".join(f"{item.score:.3f}" for item in results)
            print(f"{scores}   {question}")
