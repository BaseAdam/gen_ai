import json
import math
import re

from pydantic import BaseModel

NUMBER_PATTERN = re.compile(r"\d{1,3}(?:[,  ]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
NUMBER_TOLERANCE = 0.005


class AgentTask(BaseModel):
    id: str
    question: str
    expected_titles: list[str] = []
    forbidden_titles: list[str] = []
    expected_numbers: list[float] = []
    refuse: bool = False
    expected_tools: list[str] = []
    forbidden_tools: list[str] = []
    max_tool_calls: int | None = None
    max_steps: int | None = None


def load_tasks(path: str) -> list[AgentTask]:
    with open(path, encoding="utf-8") as f:
        raw_tasks = json.load(f)
    return [AgentTask.model_validate(raw) for raw in raw_tasks]


def numbers_in(text: str) -> list[float]:
    return [float(re.sub(r"[,  ]", "", match)) for match in NUMBER_PATTERN.findall(text)]


def check_answer(
    task: AgentTask, answer: str, all_titles: list[str], offers: list[str] | None = None
) -> list[str]:
    known = {title.lower() for title in all_titles}
    problems = []
    if offers is None:
        text = answer.lower()
        named = {title for title in known if title in text}
    else:
        named = {offer.strip().lower() for offer in offers}
        problems += [
            f"unknown offer {offer}" for offer in offers if offer.strip().lower() not in known
        ]

    problems += [f"missing {title}" for title in task.expected_titles if title.lower() not in named]
    problems += [f"unexpected {title}" for title in task.forbidden_titles if title.lower() in named]

    found = numbers_in(answer)
    for number in task.expected_numbers:
        if not any(math.isclose(value, number, rel_tol=NUMBER_TOLERANCE) for value in found):
            problems.append(f"missing number {number:g}")

    if task.refuse:
        mentioned = [title for title in all_titles if title.lower() in named]
        if mentioned:
            problems.append(f"should refuse, but mentions {', '.join(mentioned)}")
    return problems


def check_trajectory(task: AgentTask, steps: list[list[str]]) -> list[str]:
    tool_calls = [tool for step in steps for tool in step]
    problems = [f"never called {tool}" for tool in task.expected_tools if tool not in tool_calls]
    problems += [f"called {tool}" for tool in task.forbidden_tools if tool in tool_calls]
    if task.max_tool_calls is not None and len(tool_calls) > task.max_tool_calls:
        problems.append(f"{len(tool_calls)} tool calls, limit {task.max_tool_calls}")
    if task.max_steps is not None and len(steps) > task.max_steps:
        problems.append(f"{len(steps)} steps, limit {task.max_steps}")
    return problems
