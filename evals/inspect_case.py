import sys

from offers.config import Settings
from offers.evals import embed_questions_cached, load_cases
from offers.llm import create_client
from offers.models import ScoredOffer
from offers.rag import RERANK_CANDIDATES
from offers.rerank import rerank
from offers.retrieval import CANDIDATES, hybrid_search
from offers.store import connect, search_keywords, search_offers

SHOWN = 10


def section(item: ScoredOffer) -> str:
    lines = item.chunk.split("\n")
    return lines[1] if len(lines) > 1 else "(header)"


def show(label: str, ranking: list[ScoredOffer], relevant: list[str]) -> None:
    print(f"  {label}")
    if not ranking:
        print("     (empty)")
    for rank, item in enumerate(ranking, start=1):
        title = item.offer.title
        if rank <= SHOWN or title in relevant:
            marker = ">>" if title in relevant else "  "
            print(f"  {marker} {rank:>2}. {item.score:>7.4f}  {title:<28} {section(item)}")


if __name__ == "__main__":
    settings = Settings()
    client = create_client(settings)
    cases = {case.id: case for case in load_cases("evals/retrieval.json")}
    case_ids = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    with_rerank = "--rerank" in sys.argv
    selected = [cases[case_id] for case_id in case_ids]
    vectors = embed_questions_cached(
        client,
        [case.question for case in selected],
        settings.gemini_embedding_model,
        f"evals/vectors-{settings.gemini_embedding_model}.json",
    )

    with connect(settings.database_url) as conn:
        for case in selected:
            vector = vectors[case.question]
            min_score = settings.rag_min_score
            semantic = search_offers(
                conn, vector, CANDIDATES, city=case.city, remote=case.remote, min_score=min_score
            )
            keyword = search_keywords(
                conn, case.question, CANDIDATES, city=case.city, remote=case.remote
            )
            fused = hybrid_search(
                conn,
                vector,
                case.question,
                2 * CANDIDATES,
                city=case.city,
                remote=case.remote,
                min_score=min_score,
            )

            print(f"\n{case.id}: {case.question}")
            print(f"relevant: {', '.join(case.relevant)}")
            show("vectors", semantic, case.relevant)
            show("bm25", keyword, case.relevant)
            show("RRF fusion", fused, case.relevant)
            if with_rerank:
                candidates = fused[:RERANK_CANDIDATES]
                judged = rerank(
                    client,
                    case.question,
                    candidates,
                    settings.gemini_model,
                    top_k=len(candidates),
                    min_relevance=0,
                )
                show("rerank (all scores)", judged, case.relevant)
