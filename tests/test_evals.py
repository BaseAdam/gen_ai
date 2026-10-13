from pathlib import Path
from typing import Any

import pytest
from google.genai import types

from offers.evals import (
    AnswerRecord,
    cited_numbers,
    cited_titles,
    embed_questions_cached,
    invalid_citations,
    load_answers,
    load_cases,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    save_answers,
)
from offers.io import load_offers
from offers.models import Offer, RagAnswer, ScoredOffer

RETRIEVED = ["A", "B", "C", "D"]

DEV_CASES = "evals/retrieval.json"
HOLDOUT_CASES = "evals/holdout.json"


@pytest.mark.parametrize("path", [DEV_CASES, HOLDOUT_CASES])
def test_eval_cases_point_to_existing_offers(path: str) -> None:
    titles = {offer.title for offer in load_offers("offers.json")}

    for case in load_cases(path):
        assert set(case.relevant) <= titles, case.id


@pytest.mark.parametrize("path", [DEV_CASES, HOLDOUT_CASES])
def test_eval_case_ids_are_unique(path: str) -> None:
    ids = [case.id for case in load_cases(path)]

    assert len(ids) == len(set(ids))


def test_holdout_questions_are_not_copied_from_dev_set() -> None:
    dev = {case.question for case in load_cases(DEV_CASES)}
    holdout = {case.question for case in load_cases(HOLDOUT_CASES)}

    assert not dev & holdout


def test_offer_titles_are_unique() -> None:
    titles = [offer.title for offer in load_offers("offers.json")]

    assert len(titles) == len(set(titles))


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


class FakeEmbedModels:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_content(self, **kwargs: Any) -> types.EmbedContentResponse:
        texts = kwargs["contents"]
        self.calls.append(texts)
        embeddings = [types.ContentEmbedding(values=[float(len(text))]) for text in texts]
        return types.EmbedContentResponse(embeddings=embeddings)


class FakeEmbedClient:
    def __init__(self) -> None:
        self.models = FakeEmbedModels()


def test_embed_questions_cached_embeds_only_new_questions(tmp_path: Path) -> None:
    client = FakeEmbedClient()
    path = str(tmp_path / "vectors.json")

    first = embed_questions_cached(client, ["ab", "abc"], "model", path)
    second = embed_questions_cached(client, ["abc", "abcd"], "model", path)

    assert first == {"ab": [2.0], "abc": [3.0]}
    assert second == {"abc": [3.0], "abcd": [4.0]}
    assert client.models.calls == [["ab", "abc"], ["abcd"]]


def test_embed_questions_cached_skips_api_when_everything_is_cached(tmp_path: Path) -> None:
    client = FakeEmbedClient()
    path = str(tmp_path / "vectors.json")
    embed_questions_cached(client, ["ab"], "model", path)

    embed_questions_cached(client, ["ab"], "model", path)

    assert client.models.calls == [["ab"]]


def test_answers_round_trip_through_json(tmp_path: Path) -> None:
    offer = Offer(title="A", company="X", city="Warsaw")
    source = ScoredOffer(offer=offer, score=0.9, chunk="A at X\nWhat we offer\nA laptop.")
    record = AnswerRecord(
        case_id="laptop",
        question="Where will I get a laptop?",
        answer=RagAnswer(answer="Offer [1] provides a laptop.", sources=[source]),
    )
    path = str(tmp_path / "answers.json")

    save_answers([record], path)

    assert load_answers(path) == [record]


def make_answer(text: str, titles: list[str]) -> RagAnswer:
    sources = [
        ScoredOffer(offer=Offer(title=title, company="X", city="Warsaw"), score=1.0)
        for title in titles
    ]
    return RagAnswer(answer=text, sources=sources)


def test_cited_numbers_reads_single_and_grouped_citations() -> None:
    assert cited_numbers("See [1] and [2, 3]. Also [1][5].") == {1, 2, 3, 5}


def test_cited_numbers_ignores_brackets_without_numbers() -> None:
    assert cited_numbers("Skills: [python, sql], not [a1] or [].") == set()


def test_invalid_citations_are_outside_the_source_list() -> None:
    answer = make_answer("Offers [0], [2] and [3].", ["A", "B"])

    assert invalid_citations(answer) == {0, 3}


def test_cited_titles_maps_numbers_to_source_titles() -> None:
    answer = make_answer("Try [2] or [1, 2], not [7].", ["A", "B", "C"])

    assert cited_titles(answer) == {"A", "B"}
