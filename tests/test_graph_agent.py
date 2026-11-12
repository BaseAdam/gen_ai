import inspect
import json
from typing import Any

import pytest
from google.genai import errors, types
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langgraph.errors import GraphRecursionError
from pydantic import Field, SecretStr

from offers import mcp_server, tools
from offers.config import Settings
from offers.graph_agent import ask_graph, build_model
from offers.llm import build_http_options
from offers.models import Offer
from offers.observability import record_tokens, request_context
from offers.pyai_agent import ANSWER_INSTRUCTION, ANSWER_TOOL, MAX_MODEL_REQUESTS, AgentAnswer
from offers.tools import AgentDeps, SearchHit

OFFERS = [
    Offer(title="A", company="X", city="Krakow", remote=True, salary_min=10000, salary_max=14000),
    Offer(title="B", company="Y", city="Warsaw", salary_min=6000, salary_max=8000),
]


class ScriptedChat(BaseChatModel):
    responses: list[AIMessage]
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    call_kwargs: list[dict[str, Any]] = Field(default_factory=list)
    bound_tools: list[Any] = Field(default_factory=list)
    tool_choice: Any = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        self.seen.append(list(messages))
        self.call_kwargs.append(kwargs)
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])

    def bind_tools(  # type: ignore[override]
        self, tools: Any, *, tool_choice: Any = None, **kwargs: Any
    ) -> Runnable[Any, AIMessage]:
        self.bound_tools = list(tools)
        self.tool_choice = tool_choice
        return self


def make_deps() -> AgentDeps:
    settings = Settings(gemini_api_key=SecretStr("test-key"), gemini_model="m", _env_file=None)
    return AgentDeps(client=None, conn=None, settings=settings, offers=OFFERS)  # type: ignore[arg-type]


def call(*calls: tuple[str, dict[str, Any]], usage: dict[str, Any] | None = None) -> AIMessage:
    tool_calls = [
        {"name": name, "args": args, "id": f"call-{number}"}
        for number, (name, args) in enumerate(calls)
    ]
    return AIMessage(content="", tool_calls=tool_calls, usage_metadata=usage)  # type: ignore[arg-type]


def final(answer: str, offers: list[str]) -> AIMessage:
    return call((ANSWER_TOOL, {"answer": answer, "offers": offers}))


def tool_messages(messages: list[BaseMessage]) -> list[ToolMessage]:
    return [message for message in messages if isinstance(message, ToolMessage)]


def hit(title: str, relevance: float) -> SearchHit:
    return SearchHit(
        title=title,
        company="X",
        city="Krakow",
        remote=True,
        salary_min=None,
        salary_max=None,
        fragment="...",
        relevance=relevance,
    )


def test_model_gets_mcp_tools_answer_tool_and_our_http_options() -> None:
    model = ScriptedChat(responses=[final("Hi", [])])

    ask_graph(make_deps(), "Hello", model=model)

    names = [getattr(tool, "name", None) or tool["function"]["name"] for tool in model.bound_tools]
    assert names == ["list_offers", "search_job_ads", "salary_stats", ANSWER_TOOL]
    for tool in model.bound_tools[:3]:
        assert tool.description.strip() == inspect.getdoc(getattr(mcp_server, tool.name))
    assert model.tool_choice == "any"
    assert model.call_kwargs[0]["http_options"] == build_http_options(make_deps().settings)
    first_prompt = model.seen[0][0]
    assert first_prompt.content == ANSWER_INSTRUCTION


def test_graph_combines_search_with_salary_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "search_job_ads", lambda deps, query, **f: [hit("A", 9)])
    model = ScriptedChat(
        responses=[
            call(("search_job_ads", {"query": "flexible working hours"})),
            call(("salary_stats", {"titles": ["A"]})),
            final("A pays 12,000 PLN on average.", ["A"]),
        ]
    )

    with request_context("task") as context:
        output = ask_graph(make_deps(), "Average salary with flexible hours?", model=model)

    assert output == AgentAnswer(answer="A pays 12,000 PLN on average.", offers=["A"])
    [search_result] = tool_messages(model.seen[1])
    assert [found["title"] for found in json.loads(str(search_result.content))["result"]] == ["A"]
    stats_result = tool_messages(model.seen[2])[-1]
    assert json.loads(str(stats_result.content))["average"] == 12000.0
    assert context.steps == [["search_job_ads"], ["salary_stats"], []]


def test_parallel_calls_are_one_step_and_all_results_go_back() -> None:
    model = ScriptedChat(
        responses=[
            call(("salary_stats", {"city": "Krakow"}), ("salary_stats", {"city": "Warsaw"})),
            final("Krakow pays more.", []),
        ]
    )

    with request_context("task") as context:
        ask_graph(make_deps(), "Krakow or Warsaw?", model=model)

    averages = [json.loads(str(m.content))["average"] for m in tool_messages(model.seen[1])]
    assert averages == [12000.0, 7000.0]
    assert context.steps == [["salary_stats", "salary_stats"], []]


def test_tool_errors_go_back_to_model_as_text(monkeypatch: pytest.MonkeyPatch) -> None:
    def overloaded(deps: AgentDeps, query: str, **filters: Any) -> list[SearchHit]:
        raise errors.ServerError(503, {"error": {"code": 503, "message": "overloaded"}})

    monkeypatch.setattr(tools, "search_job_ads", overloaded)
    model = ScriptedChat(
        responses=[
            call(("search_job_ads", {"query": "mentoring"}), ("search_job_ads", {"city": "X"})),
            final("Search is temporarily unavailable.", []),
        ]
    )

    output = ask_graph(make_deps(), "Mentoring?", model=model)

    outage, invalid = tool_messages(model.seen[1])
    assert "search_job_ads is temporarily unavailable" in str(outage.content)
    assert "query" in str(invalid.content)
    assert output.offers == []


def test_recursion_limit_stops_a_looping_model() -> None:
    responses = [call(("list_offers", {"city": "Krakow"})) for _ in range(20)]
    model = ScriptedChat(responses=responses)

    with request_context("task") as context, pytest.raises(GraphRecursionError):
        ask_graph(make_deps(), "Loop forever", model=model)

    assert len(model.seen) == MAX_MODEL_REQUESTS
    assert len(context.steps) == MAX_MODEL_REQUESTS


def test_tokens_exclude_thinking_and_count_llm_calls_inside_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def search_with_rerank(deps: AgentDeps, query: str, **filters: Any) -> list[SearchHit]:
        record_tokens("rerank", 400, 30, 0)
        return [hit("A", 9)]

    monkeypatch.setattr(tools, "search_job_ads", search_with_rerank)
    usage = {
        "input_tokens": 1000,
        "output_tokens": 50,
        "total_tokens": 1050,
        "output_token_details": {"reasoning": 30},
    }
    model = ScriptedChat(
        responses=[
            call(("search_job_ads", {"query": "mentoring"}), usage=usage),
            call((ANSWER_TOOL, {"answer": "A", "offers": ["A"]}), usage=usage),
        ]
    )

    with request_context("task") as context:
        ask_graph(make_deps(), "Mentoring?", model=model)

    assert (context.input_tokens, context.output_tokens, context.thinking_tokens) == (2400, 70, 60)
    assert context.llm_calls == 3


def test_plain_text_reply_becomes_answer_without_offers() -> None:
    model = ScriptedChat(responses=[AIMessage(content="I can only help with job offers.")])

    output = ask_graph(make_deps(), "Weather?", model=model)

    assert output == AgentAnswer(answer="I can only help with job offers.", offers=[])


def test_build_model_uses_settings_and_low_thinking() -> None:
    deps = make_deps()
    deps.settings = deps.settings.model_copy(update={"gemini_model": "gemini-3.5-flash-lite"})

    model = build_model(deps)

    assert model.model == "gemini-3.5-flash-lite"
    assert model.google_api_key is not None
    assert model.google_api_key.get_secret_value() == "test-key"
    assert model.temperature is None
    assert model.thinking_config == {"thinking_level": types.ThinkingLevel.LOW}
