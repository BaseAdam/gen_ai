from offers.calibration import BreakKind, break_answer
from offers.evals import AnswerRecord, load_answers, save_answers

ANSWERS_PATH = "evals/retrieval-answers.json"
BROKEN_PATH = "evals/broken-answers.json"
BROKEN: list[tuple[str, BreakKind]] = [
    ("junior-remote", "wrong-citation"),
    ("pytorch", "wrong-citation"),
    ("training-budget", "wrong-citation"),
    ("django", "invented-fact"),
    ("excel", "invented-fact"),
    ("internship", "invented-fact"),
    ("equipment", "off-topic"),
    ("flexible-hours", "off-topic"),
    ("training-budget", "off-topic"),
]

if __name__ == "__main__":
    originals = {record.case_id: record for record in load_answers(ANSWERS_PATH)}

    broken = []
    for case_id, kind in BROKEN:
        original = originals[case_id]
        broken.append(
            AnswerRecord(
                case_id=f"{case_id}:{kind}",
                question=original.question,
                answer=break_answer(original.answer, kind),
            )
        )

    save_answers(broken, BROKEN_PATH)
    for record in broken:
        print(f"== {record.case_id} | {record.question}")
        print(record.answer.answer)
        print()
    print(f"Saved {len(broken)} broken answers to {BROKEN_PATH}")
