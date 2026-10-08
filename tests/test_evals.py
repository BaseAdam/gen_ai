from offers.evals import load_cases
from offers.io import load_offers

CASES_PATH = "evals/retrieval.json"


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
