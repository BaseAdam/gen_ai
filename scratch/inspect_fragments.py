from offers.config import Settings
from offers.embeddings import embed_texts
from offers.llm import create_client
from offers.models import ScoredOffer
from offers.rag import RERANK_CANDIDATES
from offers.rerank import rerank
from offers.retrieval import CANDIDATES, reciprocal_rank_fusion
from offers.store import connect, search_keywords, search_offers

QUERIES = ["mentor", "support from a mentor for juniors"]
CITY = "Poznan"


def section(item: ScoredOffer) -> str:
    names = [chunk.splitlines()[1] for chunk in item.chunk.split("\n\n") if "\n" in chunk]
    return " + ".join(names) or "?"


def positions(results: list[ScoredOffer]) -> dict[str, str]:
    return {
        item.offer.title: f"#{rank} {section(item)}" for rank, item in enumerate(results, start=1)
    }


if __name__ == "__main__":
    settings = Settings()
    client = create_client(settings)
    with connect(settings.database_url) as conn:
        for query in QUERIES:
            [vector] = embed_texts(
                client, [query], settings.gemini_embedding_model, "RETRIEVAL_QUERY"
            )
            semantic = search_offers(
                conn, vector, CANDIDATES, city=CITY, min_score=settings.rag_min_score
            )
            keyword = search_keywords(conn, query, CANDIDATES, city=CITY)
            fused = reciprocal_rank_fusion([semantic, keyword])[:RERANK_CANDIDATES]
            scored = rerank(
                client, query, fused, settings.gemini_model, top_k=len(fused), min_relevance=0
            )
            relevance = {item.offer.title: item.score for item in scored}
            vector_at, keyword_at = positions(semantic), positions(keyword)

            print(f"\nquery={query!r}, city={CITY}")
            print(f"  {'offer':<28}{'vector':<24}{'bm25':<24}{'fragments shown':<36}relevance")
            for item in fused:
                title = item.offer.title
                print(
                    f"  {title:<28}{vector_at.get(title, '-'):<24}{keyword_at.get(title, '-'):<24}"
                    f"{section(item):<36}{relevance.get(title, '-')}"
                )
