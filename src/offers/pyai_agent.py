import logging

import pydantic_ai
from google.genai import types
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ToolOutput, UsageLimits, capture_run_messages
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider

from offers.agent import AGENT_INSTRUCTION
from offers.mcp_server import create_server, provide_deps
from offers.observability import record_step, record_tokens, timed
from offers.tools import AgentDeps

MAX_MODEL_REQUESTS = 8
ANSWER_TOOL = "final_result"
ANSWER_INSTRUCTION = (
    AGENT_INSTRUCTION + " Put the exact titles of the offers that answer the question in "
    "`offers`; do not include offers you mention only to exclude them."
)

logger = logging.getLogger(__name__)

# The startup banner would land in the API and eval logs (and break JSON log lines).
pydantic_ai.BANNER_ENABLED = False


class AgentAnswer(BaseModel):
    answer: str = Field(description="The answer for the user")
    offers: list[str] = Field(
        description="Exact titles of the offers that answer the question; empty if none do"
    )


offers_agent = Agent(
    output_type=ToolOutput(AgentAnswer, name=ANSWER_TOOL),
    instructions=ANSWER_INSTRUCTION,
)


def offer_tools(deps: AgentDeps) -> MCPToolset[object]:
    return MCPToolset(create_server(provide_deps(deps)))


def build_model(deps: AgentDeps) -> GoogleModel:
    settings = GoogleModelSettings(
        google_thinking_config=types.ThinkingConfigDict(thinking_level=types.ThinkingLevel.LOW),
    )
    provider = GoogleProvider(client=deps.client)
    return GoogleModel(deps.settings.gemini_model, provider=provider, settings=settings)


def record_run(messages: list[ModelMessage]) -> None:
    responses = [message for message in messages if isinstance(message, ModelResponse)]
    for response in responses:
        calls = [
            (part.tool_name, part.args_as_dict())
            for part in response.parts
            if isinstance(part, ToolCallPart) and part.tool_name != ANSWER_TOOL
        ]
        record_step("pyai", calls)
    thinking = sum(response.usage.output_reasoning_tokens for response in responses)
    record_tokens(
        "pyai.run",
        sum(response.usage.input_tokens for response in responses),
        sum(response.usage.output_tokens for response in responses) - thinking,
        thinking,
        calls=len(responses),
    )


def ask_agent(deps: AgentDeps, question: str, model: Model | None = None) -> AgentAnswer:
    with timed("pyai.run"), capture_run_messages() as messages:
        try:
            result = offers_agent.run_sync(
                question,
                model=model or build_model(deps),
                toolsets=[offer_tools(deps)],
                usage_limits=UsageLimits(request_limit=MAX_MODEL_REQUESTS),
            )
        finally:
            record_run(messages)
    return result.output
