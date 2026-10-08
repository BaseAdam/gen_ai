from offers.config import Settings
from offers.store import CANDIDATE_CHUNKS, SEARCH_OFFERS, connect

if __name__ == "__main__":
    settings = Settings()

    with connect(settings.database_url) as conn:
        [row] = conn.execute("SELECT embedding::text FROM chunks LIMIT 1").fetchall()
        params = {
            "query": row[0],
            "top_k": 5,
            "city": None,
            "remote": None,
            "min_score": None,
            "candidates": CANDIDATE_CHUNKS,
        }

        for mode in ("on", "off"):
            conn.execute(f"SET enable_seqscan = {mode}")
            plan = conn.execute("EXPLAIN (COSTS OFF) " + SEARCH_OFFERS, params).fetchall()
            print(f"--- enable_seqscan = {mode}")
            for (line,) in plan:
                print(line[:90])
