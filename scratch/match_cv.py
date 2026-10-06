from offers.config import Settings
from offers.embeddings import embed_texts
from offers.llm import create_client
from offers.store import connect, search_offers

CV = """
Fullstack developer z trzyletnim doświadczeniem w TypeScript i React.
Uczę się Pythona, FastAPI i pracy z LLM. Znam Dockera i podstawy SQL.
Szukam pracy w Warszawie albo zdalnie.
"""


if __name__ == "__main__":
    settings = Settings()
    client = create_client(settings)

    [cv_vector] = embed_texts(client, [CV], settings.gemini_embedding_model, "RETRIEVAL_QUERY")

    with connect(settings.database_url) as conn:
        for item in search_offers(conn, cv_vector, city="Warszawa", remote=True):
            print(f"{item.score:.3f}  {item.offer}")
