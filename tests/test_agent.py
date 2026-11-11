from typing import Any

import pytest
from google.genai import types
from pydantic import SecretStr

from offers import agent
from offers.agent import AGENT_INSTRUCTION, answer_with_agent, execute_tool
from offers.config import Settings
from offers.models import Offer
from offers.tools import AgentDeps, SearchHit

OFFERS = [
    Offer(
        title="A",
        company="X",
        city="Krakow",
        remote=True,
        salary_min=10000,
        salary_max=14000,
        description="Flexible working hours.",
    ),
    Offer(title="B", company="Y", city="Krakow", salary_min=6000, salary_max=8000),
    Offer(title="C", company="Z", city="Warsaw", remote=True, salary_min=20000, salary_max=30000),
]


def call_response(name: str, **args: Any) -> types.GenerateContentResponse:
    part = types.Part(function_call=types.FunctionCall(name=name, args=args))
    content = types.Content(role="model", parts=[part])
    return types.GenerateContentResponse(candidates=[types.Candidate(content=content)])


def text_response(text: str) -> types.GenerateContentResponse:
    content = types.Content(role="model", parts=[types.Part(text=text)])
    return types.GenerateContentResponse(candidates=[types.Candidate(content=content)])


class ScriptedModels:
    def __init__(self, responses: list[types.GenerateContentResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> types.GenerateContentResponse:
        self.calls.append({**kwargs, "contents": list(kwargs["contents"])})
        return self.responses.pop(0)


class ScriptedClient:
    def __init__(self, responses: list[types.GenerateContentResponse]) -> None:
        self.models = ScriptedModels(responses)


def make_deps(client: Any = None) -> AgentDeps:
    settings = Settings(
        gemini_api_key=SecretStr("test-key"), gemini_model="chat-model", _env_file=None
    )
    return AgentDeps(client=client, conn=None, settings=settings, offers=OFFERS)  # type: ignore[arg-type]


def hit(title: str, relevance: float) -> SearchHit:
    return SearchHit(
        title=title, company="X", city="Krakow", remote=True, fragment="...", relevance=relevance
    )


def test_list_offers_tool_returns_rows_without_descriptions() -> None:
    result = execute_tool(make_deps(), "list_offers", {"city": "Krakow", "remote": True})

    assert [row["title"] for row in result["result"]] == ["A"]
    assert "description" not in result["result"][0]


def test_salary_stats_tool_accepts_titles() -> None:
    result = execute_tool(make_deps(), "salary_stats", {"titles": ["A", "C"]})

    assert result["result"]["average"] == 18500.0
    assert result["result"]["offers_count"] == 2


def test_search_job_ads_tool_passes_query_and_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []

    def fake_search(deps: AgentDeps, **kwargs: Any) -> list[SearchHit]:
        seen.append(kwargs)
        return [hit("A", 9)]

    monkeypatch.setattr(agent, "search_job_ads", fake_search)

    result = execute_tool(
        make_deps(), "search_job_ads", {"query": "flexible hours", "remote": True}
    )

    assert seen == [{"query": "flexible hours", "city": None, "remote": True}]
    assert result["result"][0]["relevance"] == 9


def test_invalid_arguments_go_back_as_error() -> None:
    result = execute_tool(make_deps(), "search_job_ads", {"city": "Krakow"})

    assert "Invalid arguments" in result["error"]


def test_unknown_tool_goes_back_as_error() -> None:
    assert execute_tool(make_deps(), "delete_offers", {}) == {
        "error": "Unknown tool: delete_offers"
    }


def test_agent_combines_search_with_salary_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent, "search_job_ads", lambda deps, **kwargs: [hit("A", 9), hit("C", 5)])
    client = ScriptedClient(
        [
            call_response("search_job_ads", query="flexible working hours"),
            call_response("salary_stats", titles=["A"]),
            text_response("A has flexible hours and pays 12,000 PLN on average."),
        ]
    )

    answer = answer_with_agent(make_deps(client), "Average salary with flexible hours?")

    assert answer == "A has flexible hours and pays 12,000 PLN on average."
    stats_turn = client.models.calls[2]["contents"][-1]
    assert stats_turn.parts[0].function_response.response["result"]["average"] == 12000.0
    config = client.models.calls[0]["config"]
    assert config.system_instruction == AGENT_INSTRUCTION
    declared = [d.name for d in config.tools[0].function_declarations]
    assert declared == ["list_offers", "search_job_ads", "salary_stats"]
