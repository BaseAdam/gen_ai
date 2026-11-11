from offers.config import Settings
from offers.embeddings import embed_texts
from offers.io import load_offers
from offers.llm import create_client
from offers.store import connect, search_offers
from offers.tools import AgentDeps, search_job_ads

QUERIES = ["mentor", "support from a mentor for juniors"]

if __name__ == "__main__":
    settings = Settings()
    client = create_client(settings)
    with connect(settings.database_url) as conn:
        deps = AgentDeps(
            client=client, conn=conn, settings=settings, offers=load_offers("offers.json")
        )
        for query in QUERIES:
            [vector] = embed_texts(
                client, [query], settings.gemini_embedding_model, "RETRIEVAL_QUERY"
            )
            vector_hits = search_offers(conn, vector, 10, city="Poznan")
            print(f"\nquery={query!r}, city=Poznan")
            print(f"  vector scores (gate {settings.rag_min_score}):")
            for item in vector_hits:
                status = "passes" if item.score >= settings.rag_min_score else "blocked"
                print(f"    {item.score:.3f} {status:<8} {item.offer.title}")
            hits = search_job_ads(deps, query, city="Poznan")
            print(f"  search_job_ads: {[(hit.title, hit.relevance) for hit in hits]}")
