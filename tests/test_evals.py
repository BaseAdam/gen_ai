import pytest

from offers.evals import load_cases, precision_at_k, recall_at_k, reciprocal_rank
from offers.io import load_offers

CASES_PATH = "evals/retrieval.json"
RETRIEVED = ["A", "B", "C", "D"]


def test_eval_cases_point_to_existing_offers() -> None:
    titles = {offer.title for offer in load_offers("offers.json")}

    for case in load_cases(CASES_PATH):
        assert set(case.relevant) <= titles, case.id


def test_offer_titles_are_unique() -> None:
    titles = [offer.title for offer in load_offers("offers.json")]

    assert len(titles) == len(set(titles))


def test_eval_case_ids_are_unique() -> None:
    ids = [case.id for case in load_cases(CASES_PATH)]

    assert len(ids) == len(set(ids))


def test_precision_counts_relevant_among_top_k() -> None:
    assert precision_at_k(RETRIEVED, ["B", "D", "X"], k=2) == 0.5


def test_precision_divides_by_what_was_returned() -> None:
    assert precision_at_k(["A"], ["A"], k=5) == 1.0


def test_precision_of_empty_result_is_zero() -> None:
    assert precision_at_k([], ["A"], k=5) == 0.0


def test_recall_counts_found_relevant_within_k() -> None:
    assert recall_at_k(RETRIEVED, ["B", "D", "X"], k=2) == pytest.approx(1 / 3)
    assert recall_at_k(RETRIEVED, ["B", "D", "X"], k=4) == pytest.approx(2 / 3)


def test_reciprocal_rank_uses_first_relevant_position() -> None:
    assert reciprocal_rank(RETRIEVED, ["C", "D"]) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_without_hits() -> None:
    assert reciprocal_rank(RETRIEVED, ["X"]) == 0.0
