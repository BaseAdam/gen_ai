import pytest

from offers.calibration import (
    INVENTED_CLAIM,
    HumanLabel,
    break_answer,
    compare,
    load_labels,
    outcome,
    swap_citations,
)
from offers.evals import load_cases
from offers.models import Offer, RagAnswer, ScoredOffer, Verdict

LABELS_PATH = "evals/human-labels.json"


def make_answer(text: str, titles: list[str]) -> RagAnswer:
    sources = [
        ScoredOffer(offer=Offer(title=title, company="Acme", city="Warsaw"), score=1.0)
        for title in titles
    ]
    return RagAnswer(answer=text, sources=sources)


def test_swap_citations_exchanges_two_numbers() -> None:
    assert swap_citations("A [1], B [2], C [3].", 1, 2) == "A [2], B [1], C [3]."


def test_swap_citations_handles_grouped_citations() -> None:
    assert swap_citations("See [1,2] and [2, 3].", 1, 2) == "See [2, 1] and [1, 3]."


def test_wrong_citation_swaps_first_two_sources_and_keeps_them() -> None:
    answer = make_answer("X [1]. Y [2].", ["A", "B"])

    broken = break_answer(answer, "wrong-citation")

    assert broken.answer == "X [2]. Y [1]."
    assert broken.sources == answer.sources


def test_invented_fact_appends_unsupported_claim() -> None:
    broken = break_answer(make_answer("X [1].", ["A"]), "invented-fact")

    assert broken.answer == f"X [1].\n\n{INVENTED_CLAIM}"


def test_off_topic_only_describes_offers() -> None:
    broken = break_answer(make_answer("X [1].", ["Dev", "QA"]), "off-topic")

    assert broken.answer == "Offer [1]: Dev at Acme in Warsaw. Offer [2]: QA at Acme in Warsaw."


def test_break_answer_leaves_original_untouched() -> None:
    answer = make_answer("X [1]. Y [2].", ["A", "B"])

    break_answer(answer, "wrong-citation")

    assert answer.answer == "X [1]. Y [2]."


def test_human_labels_point_to_dev_cases() -> None:
    case_ids = {case.id for case in load_cases("evals/retrieval.json")}

    for label in load_labels(LABELS_PATH):
        base_id = label.case_id.split(":")[0]
        assert base_id in case_ids, label.case_id


def test_human_label_ids_are_unique() -> None:
    ids = [label.case_id for label in load_labels(LABELS_PATH)]

    assert len(ids) == len(set(ids))


@pytest.mark.parametrize(
    ("human", "judge", "expected"),
    [
        (True, True, "agree"),
        (False, False, "agree"),
        (False, True, "missed"),
        (True, False, "false alarm"),
    ],
)
def test_outcome_names_each_kind_of_disagreement(human: bool, judge: bool, expected: str) -> None:
    assert outcome(human, judge) == expected


def test_compare_checks_every_criterion_separately() -> None:
    label = HumanLabel(case_id="x", grounded=True, citations_correct=False, answers_question=True)
    verdict = Verdict(
        reasoning="...", grounded=True, citations_correct=True, answers_question=False
    )

    assert compare(label, verdict) == {
        "grounded": "agree",
        "citations_correct": "missed",
        "answers_question": "false alarm",
    }
