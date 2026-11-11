import logging
from typing import Any

import pytest
from google.genai import types

from offers.tool_loop import run_tool_loop


def calls_response(*calls: tuple[str, dict[str, Any]]) -> types.GenerateContentResponse:
    parts = [
        types.Part(function_call=types.FunctionCall(name=name, args=args)) for name, args in calls
    ]
    content = types.Content(role="model", parts=parts)
    return types.GenerateContentResponse(candidates=[types.Candidate(content=content)])


def text_response(text: str) -> types.GenerateContentResponse:
    content = types.Content(role="model", parts=[types.Part(text=text)])
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


def echo(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"result": f"{name} {args}"}


def test_parallel_calls_go_back_to_model_in_one_turn() -> None:
    client = ScriptedClient(
        [
            calls_response(
                ("salary_stats", {"city": "Krakow"}), ("salary_stats", {"city": "Warsaw"})
            ),
            text_response("Krakow pays more."),
        ]
    )

    answer = run_tool_loop(client, "m", "Krakow or Warsaw?", types.GenerateContentConfig(), echo)

    assert answer == "Krakow pays more."
    last_turn = client.models.sent[1][-1]
    assert last_turn.role == "user"
    assert [part.function_response.response for part in last_turn.parts] == [
        {"result": "salary_stats {'city': 'Krakow'}"},
        {"result": "salary_stats {'city': 'Warsaw'}"},
    ]


def test_tool_calls_are_logged_with_arguments(caplog: pytest.LogCaptureFixture) -> None:
    client = ScriptedClient(
        [calls_response(("list_offers", {"city": "Krakow"})), text_response("Done.")]
    )

    with caplog.at_level(logging.INFO, logger="offers"):
        run_tool_loop(client, "m", "q", types.GenerateContentConfig(), echo, name="agent")

    assert 'agent calls list_offers({"city": "Krakow"})' in caplog.messages
    assert any(message.startswith("tool.list_offers took ") for message in caplog.messages)


def test_stops_after_max_steps() -> None:
    client = ScriptedClient([calls_response(("list_offers", {})) for _ in range(5)])

    with pytest.raises(RuntimeError):
        run_tool_loop(client, "m", "q", types.GenerateContentConfig(), echo, max_steps=2)

    assert len(client.models.sent) == 2
