from typing import Any

import pytest
from google.genai import types
from pydantic import ValidationError

from offers.judge import judge_answer
from offers.models import Offer, RagAnswer, ScoredOffer, Verdict

QUESTION = "Where will I get a laptop?"
ANSWER = RagAnswer(
    answer="Offer [1] provides a laptop.",
    sources=[
        ScoredOffer(
            offer=Offer(title="A", company="X", city="Warsaw"),
            score=1.0,
            chunk="A at X\nWhat we offer\nA laptop.",
        )
    ],
)
VERDICT = Verdict(
    reasoning="The laptop claim matches offer [1].",
    grounded=True,
    citations_correct=True,
    answers_question=False,
)


class FakeJudgeModels:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> types.GenerateContentResponse:
        self.calls.append(kwargs)
        content = types.Content(role="model", parts=[types.Part(text=self.text)])
        return types.GenerateContentResponse(candidates=[types.Candidate(content=content)])


class FakeJudgeClient:
    def __init__(self, text: str) -> None:
        self.models = FakeJudgeModels(text)


def test_judge_answer_returns_parsed_verdict() -> None:
    client = FakeJudgeClient(VERDICT.model_dump_json())

    assert judge_answer(client, QUESTION, ANSWER, "test-model") == VERDICT


def test_judge_sees_sources_question_and_answer() -> None:
    client = FakeJudgeClient(VERDICT.model_dump_json())

    judge_answer(client, QUESTION, ANSWER, "test-model")

    call = client.models.calls[0]
    assert call["model"] == "test-model"
    assert call["contents"].startswith('OFFERS:\n[1] {"title":"A"')
    assert "A laptop." in call["contents"]
    assert f"QUESTION: {QUESTION}" in call["contents"]
    assert call["contents"].endswith("ANSWER:\nOffer [1] provides a laptop.")
    assert call["config"].response_schema is Verdict
    assert call["config"].temperature is None


def test_judge_answer_fails_loudly_on_invalid_response() -> None:
    client = FakeJudgeClient("looks fine")

    with pytest.raises(ValidationError):
        judge_answer(client, QUESTION, ANSWER, "test-model")


def test_verdict_starts_with_reasoning() -> None:
    assert list(Verdict.model_fields)[0] == "reasoning"
