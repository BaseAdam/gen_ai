import sys
from statistics import fmean

from offers.evals import (
    citation_precision,
    citation_recall,
    cited_titles,
    invalid_citations,
    load_answers,
    load_cases,
    relevant_in_context,
)

CASES_PATH = "evals/holdout.json" if "--holdout" in sys.argv else "evals/retrieval.json"
ANSWERS_PATH = CASES_PATH.replace(".json", "-answers.json")


def show(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def average(values: list[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return fmean(defined) if defined else None


if __name__ == "__main__":
    cases = {case.id: case for case in load_cases(CASES_PATH)}
    precisions: list[float | None] = []
    recalls: list[float | None] = []
    invalid_count = 0

    print(f"Answers: {ANSWERS_PATH}")
    print(f"{'case':<20}{'precision':>10}{'recall':>8}  notes")
    for record in load_answers(ANSWERS_PATH):
        case = cases[record.case_id]
        answer = record.answer
        precision = citation_precision(answer, case.relevant)
        recall = citation_recall(answer, case.relevant)
        precisions.append(precision)
        recalls.append(recall)

        invalid = invalid_citations(answer)
        invalid_count += len(invalid)
        cited = cited_titles(answer)
        extra = cited - set(case.relevant)
        missed = relevant_in_context(answer, case.relevant) - cited
        notes = []
        if invalid:
            notes.append(f"invalid: {sorted(invalid)}")
        if extra:
            notes.append(f"extra: {sorted(extra)}")
        if missed:
            notes.append(f"missed: {sorted(missed)}")
        print(f"{case.id:<20}{show(precision):>10}{show(recall):>8}  {'; '.join(notes)}")

    print()
    print(f"{'average':<20}{show(average(precisions)):>10}{show(average(recalls)):>8}")
    print(f"Invalid citations: {invalid_count}")
