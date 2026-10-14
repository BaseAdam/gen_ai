import os
import sys
import time
from statistics import fmean

from google.genai import errors

from offers.config import Settings
from offers.evals import VerdictRecord, load_answers, load_verdicts, save_verdicts
from offers.judge import judge_answer
from offers.llm import create_client

PAUSE_SECONDS = 13
ANSWERS_PATH = (
    "evals/holdout-answers.json" if "--holdout" in sys.argv else "evals/retrieval-answers.json"
)


def mark(value: bool) -> str:
    return "yes" if value else "NO"


if __name__ == "__main__":
    settings = Settings()
    client = create_client(settings)
    judge_model = settings.gemini_judge_model or settings.gemini_model
    verdicts_path = ANSWERS_PATH.replace("-answers.json", f"-verdicts-{judge_model}.json")

    records = load_verdicts(verdicts_path) if os.path.exists(verdicts_path) else []
    judged = {record.case_id for record in records}
    print(f"Generator: {settings.gemini_model}, judge: {judge_model}")
    print(f"Already judged: {len(judged)}")

    for item in load_answers(ANSWERS_PATH):
        if not item.answer.sources or item.case_id in judged:
            continue
        try:
            verdict = judge_answer(client, item.question, item.answer, judge_model)
        except errors.ClientError as error:
            if error.code != 429:
                raise
            print(f"Quota exhausted ({error.message}). Run again later to continue.")
            break
        records.append(VerdictRecord(case_id=item.case_id, verdict=verdict))
        save_verdicts(records, verdicts_path)
        print(f"judged {item.case_id}")
        time.sleep(PAUSE_SECONDS)

    print()
    print(f"{'case':<20}{'grounded':>10}{'citations':>11}{'answers':>9}")
    for record in records:
        verdict = record.verdict
        print(
            f"{record.case_id:<20}{mark(verdict.grounded):>10}"
            f"{mark(verdict.citations_correct):>11}{mark(verdict.answers_question):>9}"
        )
        if not (verdict.grounded and verdict.citations_correct and verdict.answers_question):
            print(f"    {verdict.reasoning}")

    verdicts = [record.verdict for record in records]
    print()
    print(
        f"{'pass rate':<20}{fmean(v.grounded for v in verdicts):>10.2f}"
        f"{fmean(v.citations_correct for v in verdicts):>11.2f}"
        f"{fmean(v.answers_question for v in verdicts):>9.2f}"
    )
    print(f"Verdicts: {verdicts_path}")
