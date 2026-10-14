from offers.calibration import INVENTED_CLAIM, break_answer, swap_citations
from offers.models import Offer, RagAnswer, ScoredOffer


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
