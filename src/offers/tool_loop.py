from collections.abc import Callable
from typing import Any

from google import genai
from google.genai import types

from offers.observability import log_usage, record_step, timed

type ToolExecutor = Callable[[str, dict[str, Any]], dict[str, Any]]


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
        calls = [
            (call.name or "unknown", call.args or {}) for call in response.function_calls or []
        ]
        record_step(name, calls)
        if not calls:
            return response.text or "No answer"

        if response.candidates and response.candidates[0].content:
            contents.append(response.candidates[0].content)

        parts = []
        for tool, args in calls:
            with timed(f"tool.{tool}"):
                result = execute(tool, args)
            parts.append(types.Part.from_function_response(name=tool, response=result))
        contents.append(types.Content(role="user", parts=parts))

    raise RuntimeError("The model did not finish within the allowed number of steps")
