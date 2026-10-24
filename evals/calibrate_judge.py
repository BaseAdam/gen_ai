import os
from collections import Counter

from offers.calibration import CRITERIA, compare, load_labels
from offers.config import Settings
from offers.evals import load_verdicts
from offers.models import Verdict

LABELS_PATH = "evals/human-labels.json"
ANSWER_SETS = ["retrieval", "broken"]


def show(outcome: str) -> str:
    return "ok" if outcome == "agree" else outcome.upper()


if __name__ == "__main__":
    settings = Settings()
    judge_model = settings.gemini_judge_model or settings.gemini_model

    verdicts: dict[str, Verdict] = {}
    for name in ANSWER_SETS:
        path = f"evals/{name}-verdicts-{judge_model}.json"
        if os.path.exists(path):
            verdicts.update({record.case_id: record.verdict for record in load_verdicts(path)})

    counts = {criterion: Counter[str]() for criterion in CRITERIA}
    pending = []
    print(f"Judge: {judge_model}")
    print(f"{'case':<32}{'grounded':>13}{'citations':>13}{'answers':>13}")
    for label in load_labels(LABELS_PATH):
        verdict = verdicts.get(label.case_id)
        if verdict is None:
            pending.append(label.case_id)
            continue
        outcomes = compare(label, verdict)
        for criterion, result in outcomes.items():
            counts[criterion][result] += 1
        cells = "".join(f"{show(outcomes[criterion]):>13}" for criterion in CRITERIA)
        print(f"{label.case_id:<32}{cells}")
        if any(result != "agree" for result in outcomes.values()):
            print(f"    {verdict.reasoning}")

    print()
    print(f"{'criterion':<20}{'agree':>8}{'missed':>8}{'false alarms':>14}")
    for criterion in CRITERIA:
        count = counts[criterion]
        total = count.total()
        print(
            f"{criterion:<20}{count['agree']:>5}/{total:<2}"
            f"{count['missed']:>8}{count['false alarm']:>14}"
        )
    if pending:
        print(f"\nWaiting for verdicts: {', '.join(pending)}")
