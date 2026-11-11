import json
import logging
from collections.abc import Callable
from typing import Any

from google import genai
from google.genai import types

from offers.observability import log_usage, timed

type ToolExecutor = Callable[[str, dict[str, Any]], dict[str, Any]]

logger = logging.getLogger(__name__)


def run_tool_loop(
    client: genai.Client,
    model: str,
    question: str,
    config: types.GenerateContentConfig,
    execute: ToolExecutor,
    max_steps: int = 5,
    name: str = "agent",
) -> str:
    contents: list[types.Content] = [types.Content(role="user", parts=[types.Part(text=question)])]

    for _ in range(max_steps):
        response = client.models.generate_content(model=model, contents=contents, config=config)
        log_usage(f"{name}.step", response)
        calls = response.function_calls
        if not calls:
            return response.text or "No answer"

        if response.candidates and response.candidates[0].content:
            contents.append(response.candidates[0].content)

        parts = []
        for call in calls:
            tool = call.name or "unknown"
            args = call.args or {}
            logger.info(
                "%s calls %s(%s)", name, tool, json.dumps(args, ensure_ascii=False, default=str)
            )
            with timed(f"tool.{tool}"):
                result = execute(tool, args)
            parts.append(types.Part.from_function_response(name=tool, response=result))
        contents.append(types.Content(role="user", parts=parts))

    raise RuntimeError("The model did not finish within the allowed number of steps")
