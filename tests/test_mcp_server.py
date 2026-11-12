import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from google.genai import errors
from mcp import Client
from mcp.types import CallToolResult, ListToolsResult, TextContent
from pydantic import SecretStr

from offers import mcp_server, tools
from offers.config import Settings
from offers.mcp_server import create_server, provide_deps
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
        skills=["python"],
    ),
    Offer(title="B", company="Y", city="Krakow", salary_min=6000, salary_max=8000),
    Offer(title="C", company="Z", city="Warsaw", remote=True, salary_min=20000, salary_max=30000),
]


def make_settings() -> Settings:
    return Settings(gemini_api_key=SecretStr("test-key"), gemini_model="m", _env_file=None)


def make_deps() -> AgentDeps:
    return AgentDeps(client=None, conn=None, settings=make_settings(), offers=OFFERS)  # type: ignore[arg-type]


def list_tools() -> ListToolsResult:
    async def scenario() -> ListToolsResult:
        async with Client(create_server(provide_deps(make_deps()))) as client:
            return await client.list_tools()

    return asyncio.run(scenario())


def call_tool(name: str, args: dict[str, Any]) -> CallToolResult:
    async def scenario() -> CallToolResult:
        async with Client(create_server(provide_deps(make_deps()))) as client:
            return await client.call_tool(name, args)

    return asyncio.run(scenario())


def error_text(result: CallToolResult) -> str:
    assert result.is_error
    [content] = result.content
    assert isinstance(content, TextContent)
    return content.text


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


def test_server_declares_the_three_tools_with_described_parameters() -> None:
    declared = {tool.name: tool for tool in list_tools().tools}

    assert list(declared) == ["list_offers", "search_job_ads", "salary_stats"]
    search = declared["search_job_ads"]
    assert search.description is not None
    assert search.description.startswith("Searches the text of job ads by meaning")
    assert search.input_schema["required"] == ["query"]
    assert list(search.input_schema["properties"]) == ["query", "city", "remote"]
    for tool in declared.values():
        for name, schema in tool.input_schema["properties"].items():
            assert schema.get("description"), f"{tool.name}.{name} has no description"
    stats_output = declared["salary_stats"].output_schema
    assert stats_output is not None
    assert "average" in stats_output["properties"]


def test_list_offers_returns_structured_rows() -> None:
    result = call_tool("list_offers", {"city": "Krakow", "remote": True})

    assert not result.is_error
    assert result.structured_content == {
        "total": 1,
        "offers": [
            {
                "title": "A",
                "company": "X",
                "city": "Krakow",
                "remote": True,
                "salary_min": 10000,
                "salary_max": 14000,
                "skills": ["python"],
            }
        ],
    }


@pytest.mark.parametrize(
    ("args", "titles"),
    [
        ({"city": "Krakow"}, ["A", "B"]),
        ({"remote": True}, ["A", "C"]),
        ({"skill": "python"}, ["A"]),
    ],
)
def test_list_offers_passes_every_filter(args: dict[str, Any], titles: list[str]) -> None:
    result = call_tool("list_offers", args)

    assert result.structured_content is not None
    assert [offer["title"] for offer in result.structured_content["offers"]] == titles


@pytest.mark.parametrize(
    ("args", "average"),
    [
        ({"titles": ["A", "C"]}, 18500.0),
        ({"city": "Krakow"}, 9500.0),
        ({"remote": True}, 18500.0),
        ({"skill": "python"}, 12000.0),
    ],
)
def test_salary_stats_passes_every_condition(args: dict[str, Any], average: float) -> None:
    result = call_tool("salary_stats", args)

    assert result.structured_content is not None
    assert result.structured_content["average"] == average


def test_search_job_ads_passes_query_and_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    searches: list[dict[str, Any]] = []

    def fake_search(deps: AgentDeps, query: str, **filters: Any) -> list[SearchHit]:
        searches.append({"query": query, **filters})
        return [hit("A", 9)]

    monkeypatch.setattr(tools, "search_job_ads", fake_search)

    result = call_tool(
        "search_job_ads", {"query": "mentoring for juniors", "city": "Poznan", "remote": True}
    )

    assert searches == [{"query": "mentoring for juniors", "city": "Poznan", "remote": True}]
    assert result.structured_content is not None
    assert [found["title"] for found in result.structured_content["result"]] == ["A"]


def test_search_outage_goes_back_to_client_as_tool_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def overloaded(deps: AgentDeps, query: str, **filters: Any) -> list[SearchHit]:
        raise errors.ServerError(503, {"error": {"code": 503, "message": "overloaded"}})

    monkeypatch.setattr(tools, "search_job_ads", overloaded)

    result = call_tool("search_job_ads", {"query": "mentoring for juniors"})

    assert "search_job_ads is temporarily unavailable" in error_text(result)


def test_unexpected_crash_does_not_leak_details(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(deps: AgentDeps, query: str, **filters: Any) -> list[SearchHit]:
        raise RuntimeError("password=secret in connection string")

    monkeypatch.setattr(tools, "search_job_ads", broken)

    result = call_tool("search_job_ads", {"query": "mentoring for juniors"})

    assert error_text(result) == "Error executing tool search_job_ads"


def test_invalid_arguments_go_back_as_error() -> None:
    result = call_tool("search_job_ads", {"city": "Krakow"})

    assert "query" in error_text(result)


def test_default_lifespan_opens_deps_and_closes_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    @contextmanager
    def fake_connect(url: str) -> Iterator[str]:
        events.append(f"open {url}")
        yield "conn"
        events.append("close")

    monkeypatch.setattr(mcp_server, "Settings", make_settings)
    monkeypatch.setattr(mcp_server, "connect", fake_connect)
    monkeypatch.setattr(mcp_server, "create_client", lambda settings: "client")
    monkeypatch.setattr(mcp_server, "load_offers", lambda path: OFFERS)

    async def scenario() -> CallToolResult:
        async with Client(create_server()) as client:
            return await client.call_tool("list_offers", {"city": "Warsaw"})

    result = asyncio.run(scenario())

    assert result.structured_content is not None
    assert result.structured_content["total"] == 1
    assert events == [f"open {make_settings().database_url}", "close"]
