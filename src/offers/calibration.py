import re
from typing import Literal

from offers.evals import CITATION_PATTERN
from offers.models import RagAnswer

type BreakKind = Literal["wrong-citation", "invented-fact", "off-topic"]

INVENTED_CLAIM = "On top of that, this employer offers a company car and a four-day working week."


def swap_citations(text: str, first: int, second: int) -> str:
    swapped = {first: second, second: first}

    def replace(match: re.Match[str]) -> str:
        numbers = [int(part) for part in match.group(1).split(",")]
        return "[" + ", ".join(str(swapped.get(number, number)) for number in numbers) + "]"

    return CITATION_PATTERN.sub(replace, text)


def describe_offers(answer: RagAnswer) -> str:
    lines = [
        f"Offer [{number}]: {source.offer.title} at {source.offer.company} in {source.offer.city}."
        for number, source in enumerate(answer.sources, start=1)
    ]
    return " ".join(lines)


def break_answer(answer: RagAnswer, kind: BreakKind) -> RagAnswer:
    if kind == "wrong-citation":
        text = swap_citations(answer.answer, 1, 2)
    elif kind == "invented-fact":
        text = f"{answer.answer}\n\n{INVENTED_CLAIM}"
    else:
        text = describe_offers(answer)
    return answer.model_copy(update={"answer": text})
