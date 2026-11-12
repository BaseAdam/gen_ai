from offers.config import eval_settings
from offers.evals import embed_questions_cached, load_cases
from offers.llm import create_client
from offers.store import connect, search_offers

if __name__ == "__main__":
    settings = eval_settings()
    client = create_client(settings)
    cases = load_cases("evals/retrieval.json")
    vectors = embed_questions_cached(
        client,
        [case.question for case in cases],
        settings.gemini_embedding_model,
        f"evals/vectors-{settings.gemini_embedding_model}.json",
    )

    rows = []
    with connect(settings.database_url) as conn:
        for case in cases:
            vector = vectors[case.question]
            [best] = search_offers(conn, vector, top_k=1, city=case.city, remote=case.remote)
            rows.append((best.score, case))

    print(f"min_score = {settings.rag_min_score}")
    for score, case in sorted(rows, key=lambda row: row[0], reverse=True):
        kind = "positive" if case.relevant else "NEGATIVE"
        verdict = "passes" if score >= settings.rag_min_score else "blocked"
        print(f"{score:.3f}  {kind:<8}  {verdict:<7}  {case.id}")
