from typing import Any

import pytest
from google.genai import types

from offers.assistant import answer_question, run_tool
from offers.models import Offer


def text_response(text: str) -> types.GenerateContentResponse:
    part = types.Part(text=text)
    content = types.Content(role="model", parts=[part])
    return types.GenerateContentResponse(candidates=[types.Candidate(content=content)])


def call_response(name: str, **args: Any) -> types.GenerateContentResponse:
    part = types.Part(function_call=types.FunctionCall(name=name, args=args))
    content = types.Content(role="model", parts=[part])
    return types.GenerateContentResponse(candidates=[types.Candidate(content=content)])


class ScriptedModels:
    def __init__(self, responses: list[types.GenerateContentResponse]) -> None:
        self.responses = responses
        self.sent: list[list[types.Content]] = []

    def generate_content(self, **kwargs: Any) -> types.GenerateContentResponse:
        self.sent.append(list(kwargs["contents"]))
        return self.responses.pop(0)


class ScriptedClient:
    def __init__(self, responses: list[types.GenerateContentResponse]) -> None:
        self.models = ScriptedModels(responses)


@pytest.fixture
def offers() -> list[Offer]:
    return [
        Offer(title="A", company="X", city="Warsaw", salary_min=10000, salary_max=20000),
        Offer(title="B", company="Y", city="Warsaw", salary_min=8000, salary_max=12000),
        Offer(title="C", company="Z", city="Krakow", remote=True),
    ]


def tool_result(client: ScriptedClient) -> dict[str, Any]:
    last_turn = client.models.sent[-1][-1]
    return last_turn.parts[0].function_response.response


def test_answers_directly_without_tools(offers: list[Offer]) -> None:
    client = ScriptedClient([text_response("Hi!")])

    answer = answer_question(client, "hey", offers, "test-model")

    assert answer == "Hi!"
    assert len(client.models.sent) == 1


def test_runs_tool_and_returns_final_answer(offers: list[Offer]) -> None:
    client = ScriptedClient(
        [call_response("search_offers", city="Warsaw"), text_response("There are 2 offers.")]
    )

    answer = answer_question(client, "offers in Warsaw?", offers, "test-model")

    assert answer == "There are 2 offers."
    assert len(client.models.sent) == 2
    assert [turn.role for turn in client.models.sent[1]] == ["user", "model", "user"]
    assert [o["title"] for o in tool_result(client)["result"]] == ["A", "B"]


def test_unknown_tool_goes_back_to_model_as_error(offers: list[Offer]) -> None:
    client = ScriptedClient([call_response("delete_everything"), text_response("I can't do that.")])

    answer = answer_question(client, "delete everything", offers, "test-model")

    assert answer == "I can't do that."
    assert "Unknown tool" in tool_result(client)["error"]


def test_invalid_args_go_back_to_model_as_error(offers: list[Offer]) -> None:
    client = ScriptedClient(
        [call_response("search_offers", remote="maybe"), text_response("Please clarify.")]
    )

    answer_question(client, "remote?", offers, "test-model")

    assert "Invalid arguments" in tool_result(client)["error"]


def test_stops_after_max_steps(offers: list[Offer]) -> None:
    client = ScriptedClient([call_response("search_offers") for _ in range(10)])

    with pytest.raises(RuntimeError):
        answer_question(client, "loop", offers, "test-model", max_steps=3)

    assert len(client.models.sent) == 3


def test_salary_stats_tool(offers: list[Offer]) -> None:
    result = run_tool("salary_stats", {"city": "Warsaw"}, offers)

    assert result == {"result": {"offers_count": 2, "average": 12500.0, "min": 8000, "max": 20000}}
