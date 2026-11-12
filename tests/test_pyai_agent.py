import inspect
import logging
from collections.abc import Callable
from typing import Any

import pytest
from google import genai
from google.genai import errors
from pydantic_ai import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from offers import mcp_server, tools
from offers.agent import AGENT_INSTRUCTION
from offers.models import Offer
from offers.observability import record_tokens, request_context
from offers.pyai_agent import (
    ANSWER_TOOL,
    MAX_MODEL_REQUESTS,
    AgentAnswer,
    ask_agent,
    build_model,
)
from offers.tools import AgentDeps, SearchHit
from tests.helpers import hit, make_deps

OFFERS = [
    Offer(title="A", company="X", city="Krakow", remote=True, salary_min=10000, salary_max=14000),
    Offer(title="B", company="Y", city="Krakow", salary_min=6000, salary_max=8000),
]

type Respond = Callable[[list[ModelMessage], AgentInfo], ModelResponse]


def call(tool: str, **args: Any) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool, args)])


def final(answer: str, offers: list[str]) -> ModelResponse:
    return call(ANSWER_TOOL, answer=answer, offers=offers)


def scripted(*responses: ModelResponse) -> tuple[FunctionModel, list[list[ModelMessage]]]:
    queue = list(responses)
    seen: list[list[ModelMessage]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(list(messages))
        return queue.pop(0)

    return FunctionModel(respond), seen


def last_request_parts(messages: list[ModelMessage]) -> list[Any]:
    request = messages[-1]
    assert isinstance(request, ModelRequest)
    return list(request.parts)


def test_tools_come_from_mcp_server_and_output_from_code() -> None:
    infos: list[AgentInfo] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        infos.append(info)
        return final("Hi", [])

    ask_agent(make_deps(OFFERS), "Hello", model=FunctionModel(respond))

    info = infos[0]
    assert info.instructions is not None and info.instructions.startswith(AGENT_INSTRUCTION)
    declared = {tool.name: tool for tool in info.function_tools}
    assert list(declared) == ["list_offers", "search_job_ads", "salary_stats"]
    search = declared["search_job_ads"].parameters_json_schema
    assert search["required"] == ["query"]
    assert list(search["properties"]) == ["query", "city", "remote"]
    assert search["properties"]["query"]["description"].startswith("A short phrase")
    for name, tool in declared.items():
        assert (tool.description or "").strip() == inspect.getdoc(getattr(mcp_server, name)), name
    output = info.output_tools[0]
    assert output.name == ANSWER_TOOL == "final_result"
    assert output.parameters_json_schema["required"] == ["answer", "offers"]
    assert info.allow_text_output is False


def test_agent_combines_search_with_salary_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    searches: list[dict[str, Any]] = []

    def fake_search(deps: AgentDeps, query: str, **filters: Any) -> list[SearchHit]:
        searches.append({"query": query, **filters})
        return [hit("A", 9), hit("B", 5)]

    monkeypatch.setattr(tools, "search_job_ads", fake_search)
    model, seen = scripted(
        call("search_job_ads", query="flexible working hours", remote=True),
        call("salary_stats", titles=["A"]),
        final("A has flexible hours and pays 12,000 PLN on average.", ["A"]),
    )

    output = ask_agent(make_deps(OFFERS), "Average salary with flexible hours?", model=model)

    assert output == AgentAnswer(
        answer="A has flexible hours and pays 12,000 PLN on average.", offers=["A"]
    )
    assert searches == [{"query": "flexible working hours", "city": None, "remote": True}]
    [search_return] = last_request_parts(seen[1])
    assert isinstance(search_return, ToolReturnPart)
    assert [found["title"] for found in search_return.content] == ["A", "B"]
    [stats_return] = last_request_parts(seen[2])
    assert isinstance(stats_return, ToolReturnPart)
    assert stats_return.content["average"] == 12000.0


def test_search_outage_goes_back_to_model_for_one_more_try(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def overloaded(deps: AgentDeps, query: str, **filters: Any) -> list[SearchHit]:
        raise errors.ServerError(503, {"error": {"code": 503, "message": "overloaded"}})

    monkeypatch.setattr(tools, "search_job_ads", overloaded)
    model, seen = scripted(
        call("search_job_ads", query="mentoring for juniors"),
        final("Search is temporarily unavailable, please try again later.", []),
    )

    output = ask_agent(make_deps(OFFERS), "Where will a junior get a mentor?", model=model)

    assert output.offers == []
    [retry] = last_request_parts(seen[1])
    assert isinstance(retry, ToolReturnPart)
    assert retry.outcome == "retried"
    assert "search_job_ads is temporarily unavailable" in retry.model_response_str()


def test_unexpected_tool_crash_reaches_model_without_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(deps: AgentDeps, query: str, **filters: Any) -> list[SearchHit]:
        raise RuntimeError("password=secret")

    monkeypatch.setattr(tools, "search_job_ads", broken)
    model, seen = scripted(
        call("search_job_ads", query="mentoring for juniors"),
        final("Search failed, please try again later.", []),
    )

    ask_agent(make_deps(OFFERS), "Where will a junior get a mentor?", model=model)

    [retry] = last_request_parts(seen[1])
    assert isinstance(retry, ToolReturnPart)
    assert retry.outcome == "retried"
    assert "Error executing tool search_job_ads" in retry.model_response_str()
    assert "secret" not in retry.model_response_str()


def test_llm_calls_inside_tools_are_counted_in_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    def search_with_rerank(deps: AgentDeps, query: str, **filters: Any) -> list[SearchHit]:
        record_tokens("rerank", 400, 30, 0)
        return [hit("A", 9)]

    monkeypatch.setattr(tools, "search_job_ads", search_with_rerank)
    model, _ = scripted(call("search_job_ads", query="mentoring"), final("A.", ["A"]))

    with request_context("task") as context:
        ask_agent(make_deps(OFFERS), "Mentoring?", model=model)

    assert context.llm_calls == 3
    assert len(context.steps) == 2


def test_second_search_outage_ends_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    def overloaded(deps: AgentDeps, query: str, **filters: Any) -> list[SearchHit]:
        raise errors.ServerError(503, {"error": {"code": 503, "message": "overloaded"}})

    monkeypatch.setattr(tools, "search_job_ads", overloaded)
    model, _ = scripted(
        call("search_job_ads", query="mentoring for juniors"),
        call("search_job_ads", query="mentor for a junior"),
    )

    with pytest.raises(UnexpectedModelBehavior, match="exceeded max retries"):
        ask_agent(make_deps(OFFERS), "Where will a junior get a mentor?", model=model)


def test_invalid_tool_arguments_go_back_to_model() -> None:
    model, seen = scripted(
        call("search_job_ads", city="Krakow"),
        final("Sorry.", []),
    )

    ask_agent(make_deps(OFFERS), "Anything in Krakow?", model=model)

    [retry] = last_request_parts(seen[1])
    assert isinstance(retry, ToolReturnPart)
    assert retry.outcome == "retried"
    assert "Field required" in retry.model_response_str()


def test_request_limit_stops_a_looping_model() -> None:
    calls: list[int] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(1)
        return call("list_offers", city="Krakow")

    with request_context("task") as context, pytest.raises(UsageLimitExceeded):
        ask_agent(make_deps(OFFERS), "Loop forever", model=FunctionModel(respond))

    assert len(calls) == MAX_MODEL_REQUESTS
    assert context.llm_calls == MAX_MODEL_REQUESTS
    assert context.steps == [["list_offers"]] * MAX_MODEL_REQUESTS


def test_ask_agent_logs_tool_calls_and_counts_usage(caplog: pytest.LogCaptureFixture) -> None:
    usage = RequestUsage(input_tokens=1000, output_tokens=50, output_reasoning_tokens=30)
    first = ModelResponse(parts=[ToolCallPart("salary_stats", {"titles": ["A"]})], usage=usage)
    second = ModelResponse(parts=final("A pays 12,000 PLN.", ["A"]).parts, usage=usage)
    model, _ = scripted(first, second)
    with request_context("task") as context, caplog.at_level(logging.INFO, logger="offers"):
        ask_agent(make_deps(OFFERS), "How much does A pay?", model=model)

    assert 'pyai calls salary_stats({"titles": ["A"]})' in caplog.messages
    assert "pyai.run tokens: input=2000 output=40 thinking=60" in caplog.messages
    assert (context.input_tokens, context.output_tokens, context.thinking_tokens) == (2000, 40, 60)
    assert context.llm_calls == 2
    assert context.steps == [["salary_stats"], []]
    assert not any(ANSWER_TOOL in message for message in caplog.messages)


def test_build_model_uses_shared_client_and_deterministic_settings() -> None:
    client = genai.Client(api_key="test-key")

    model = build_model(make_deps(OFFERS, client))

    assert model.model_name == "chat-model"
    assert model.settings is not None
    assert "temperature" not in model.settings
    thinking = model.settings.get("google_thinking_config")
    assert thinking is not None and thinking.get("thinking_level") == "LOW"
