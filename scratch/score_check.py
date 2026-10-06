from offers.config import Settings
from offers.embeddings import embed_texts
from offers.llm import create_client
from offers.store import connect, search_offers

QUESTIONS = [
    "Szukam pracy jako programista Python",
    "Oferty dla frontend developera z Reactem",
    "Praca zdalna przy machine learningu",
    "Praca dla kierowcy ciężarówki",
    "Przepis na pierogi ukraińskie",
    "Jaka jest stolica Francji?",
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
