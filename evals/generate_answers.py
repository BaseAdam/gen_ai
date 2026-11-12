import sys
import time

from offers.config import eval_settings
from offers.evals import AnswerRecord, load_cases, save_answers
from offers.llm import create_client
from offers.rag import answer_with_rag
from offers.store import connect

PAUSE_SECONDS = 8
CASES_PATH = "evals/holdout.json" if "--holdout" in sys.argv else "evals/retrieval.json"
ANSWERS_PATH = CASES_PATH.replace(".json", "-answers.json")

if __name__ == "__main__":
    settings = eval_settings()
    client = create_client(settings)

    records = []
    with connect(settings.database_url) as conn:
        for case in load_cases(CASES_PATH):
            result = answer_with_rag(
                client,
                conn,
                case.question,
                settings.gemini_model,
                settings.gemini_embedding_model,
                city=case.city,
                remote=case.remote,
                min_score=settings.rag_min_score,
                use_rerank=settings.rag_rerank,
            )
            records.append(AnswerRecord(case_id=case.id, question=case.question, answer=result))
            print(f"{case.id:<20} {len(result.sources)} sources  {result.answer[:60]!r}")
            time.sleep(PAUSE_SECONDS)

    save_answers(records, ANSWERS_PATH)
    print(f"Saved {len(records)} answers to {ANSWERS_PATH}")
