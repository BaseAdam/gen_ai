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


def load_tasks(path: str) -> list[AgentTask]:
    with open(path, encoding="utf-8") as f:
        raw_tasks = json.load(f)
    return [AgentTask.model_validate(raw) for raw in raw_tasks]


def numbers_in(text: str) -> list[float]:
    return [float(re.sub(r"[,  ]", "", match)) for match in NUMBER_PATTERN.findall(text)]


def check_answer(task: AgentTask, answer: str, all_titles: list[str]) -> list[str]:
    text = answer.lower()
    problems = [f"missing {title}" for title in task.expected_titles if title.lower() not in text]
    problems += [f"unexpected {title}" for title in task.forbidden_titles if title.lower() in text]

    found = numbers_in(answer)
    for number in task.expected_numbers:
        if not any(math.isclose(value, number, rel_tol=NUMBER_TOLERANCE) for value in found):
            problems.append(f"missing number {number:g}")

    if task.refuse:
        mentioned = [title for title in all_titles if title.lower() in text]
        if mentioned:
            problems.append(f"should refuse, but mentions {', '.join(mentioned)}")
    return problems
