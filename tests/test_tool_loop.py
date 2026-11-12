import logging
from typing import Any

import pytest
from google.genai import types

from offers.observability import request_context
from offers.tool_loop import run_tool_loop
from tests.helpers import ScriptedClient, calls_response, text_response


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


def test_steps_are_recorded_in_current_request() -> None:
    client = ScriptedClient(
        [
            calls_response(
                ("salary_stats", {"city": "Krakow"}), ("salary_stats", {"city": "Warsaw"})
            ),
            calls_response(("list_offers", {})),
            text_response("Done."),
        ]
    )

    with request_context("task") as context:
        run_tool_loop(client, "m", "q", types.GenerateContentConfig(), echo)

    assert context.steps == [["salary_stats", "salary_stats"], ["list_offers"], []]
    assert context.tool_calls == ["salary_stats", "salary_stats", "list_offers"]


def test_stops_after_max_steps() -> None:
    client = ScriptedClient([calls_response(("list_offers", {})) for _ in range(5)])

    with pytest.raises(RuntimeError):
        run_tool_loop(client, "m", "q", types.GenerateContentConfig(), echo, max_steps=2)

    assert len(client.models.sent) == 2
