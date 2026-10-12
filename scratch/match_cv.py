from offers.config import Settings
from offers.embeddings import embed_texts
from offers.llm import create_client
from offers.store import connect, search_offers

CV = """
Fullstack developer with three years of experience in TypeScript and React.
Learning Python, FastAPI and working with LLMs. I know Docker and the basics of SQL.
Looking for a job in Warsaw or remote.
"""


if __name__ == "__main__":
    settings = Settings()
    client = create_client(settings)

    [cv_vector] = embed_texts(client, [CV], settings.gemini_embedding_model, "RETRIEVAL_QUERY")

    with connect(settings.database_url) as conn:
        for item in search_offers(conn, cv_vector, city="Warsaw", remote=True):
            print(f"{item.score:.3f}  {item.offer}")
