import json
import logging

import httpx
from google.genai import errors, types
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext, UsageLimits
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider

from offers import tools
from offers.agent import AGENT_INSTRUCTION
from offers.observability import record_tokens, timed
from offers.tools import AgentDeps, OfferList, SalaryStats, SearchHit

MAX_MODEL_REQUESTS = 8

logger = logging.getLogger(__name__)


class AgentAnswer(BaseModel):
    answer: str = Field(description="The answer for the user")
    offers: list[str] = Field(
        description="Exact titles of the offers that answer the question; empty if none do"
    )


offers_agent = Agent(
    deps_type=AgentDeps,
    output_type=AgentAnswer,
    instructions=AGENT_INSTRUCTION
    + " Put the exact titles of the offers that answer the question in `offers`; "
    "do not include offers you mention only to exclude them.",
)


@offers_agent.tool
def list_offers(
    ctx: RunContext[AgentDeps],
    city: str | None = None,
    remote: bool | None = None,
    skill: str | None = None,
) -> OfferList:
    """Lists job offers that match structured filters: city, remote work, required technology.

    Returns the total number of matches and the offers (titles, companies, cities, salaries,
    skills) without the text of the ads. Long lists are cut to the first 25.

    Args:
        city: City, e.g. Krakow
        remote: true = remote offers only, false = on-site offers only
        skill: A single technology, e.g. docker
    """
    return tools.list_offers(ctx.deps, city=city, remote=remote, skill=skill)


@offers_agent.tool
def search_job_ads(
    ctx: RunContext[AgentDeps],
    query: str,
    city: str | None = None,
    remote: bool | None = None,
) -> list[SearchHit]:
    """Searches the text of job ads by meaning: benefits, perks, culture, responsibilities.

    Use it for working conditions such as 'training budget' or 'mentoring for juniors'.
    Returns matching offers with their salary range, the fragment that matched and a relevance
    score from 0 to 10: 7 or more = the fragment answers directly, 5-6 = only a partial or
    indirect match.

    Args:
        query: A short phrase in the user's own words, e.g. 'support from a mentor for juniors'.
            Do not shorten it to a single keyword.
        city: City, e.g. Poznan
        remote: true = remote offers only, false = on-site offers only
    """
    try:
        return tools.search_job_ads(ctx.deps, query, city=city, remote=remote)
    except (errors.APIError, httpx.TimeoutException) as error:
        raise ModelRetry(f"search_job_ads is temporarily unavailable: {error}") from error


@offers_agent.tool
def salary_stats(
    ctx: RunContext[AgentDeps],
    titles: list[str] | None = None,
    city: str | None = None,
    remote: bool | None = None,
    skill: str | None = None,
) -> SalaryStats:
    """Computes salary statistics in PLN for the offers that match ALL given conditions.

    Returns the average of salary midpoints, the minimum and the maximum. To compute
    statistics for offers you have already selected, pass their exact titles.

    Args:
        titles: Exact offer titles from list_offers or search_job_ads
        city: City, e.g. Warsaw
        remote: true = remote offers only
        skill: A single technology, e.g. python
    """
    return tools.salary_stats(ctx.deps, titles=titles, city=city, remote=remote, skill=skill)


def build_model(deps: AgentDeps) -> GoogleModel:
    settings = GoogleModelSettings(
        temperature=0,
        google_thinking_config=types.ThinkingConfigDict(thinking_level=types.ThinkingLevel.LOW),
    )
    provider = GoogleProvider(client=deps.client)
    return GoogleModel(deps.settings.gemini_model, provider=provider, settings=settings)


def log_tool_calls(messages: list[ModelMessage]) -> None:
    for message in messages:
        if not isinstance(message, ModelResponse):
            continue
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                args = json.dumps(part.args_as_dict(), ensure_ascii=False, default=str)
                logger.info("pyai calls %s(%s)", part.tool_name, args)


def ask_agent(deps: AgentDeps, question: str, model: Model | None = None) -> AgentAnswer:
    with timed("pyai.run"):
        result = offers_agent.run_sync(
            question,
            deps=deps,
            model=model or build_model(deps),
            usage_limits=UsageLimits(request_limit=MAX_MODEL_REQUESTS),
        )
    log_tool_calls(result.all_messages())
    usage = result.usage
    logger.info("pyai.run: %d requests, %d tool calls", usage.requests, usage.tool_calls)
    record_tokens(
        "pyai.run",
        usage.input_tokens,
        usage.output_tokens - usage.output_reasoning_tokens,
        usage.output_reasoning_tokens,
        calls=usage.requests,
    )
    return result.output
