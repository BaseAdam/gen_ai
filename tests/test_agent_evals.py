import pytest

from offers.agent_evals import AgentTask, check_answer, load_tasks, numbers_in
from offers.io import load_offers

TASKS_PATH = "evals/agent_tasks.json"
TITLES = ["Platform Engineer", "LLM Engineer", "Backend Engineer"]


def test_numbers_in_reads_separated_thousands_and_decimals() -> None:
    text = "There are 10 offers, averaging 15,750 PLN (or 12 250, or 15,875.5)."

    assert numbers_in(text) == [10, 15750, 12250, 15875.5]


def test_check_answer_passes_when_everything_is_there() -> None:
    task = AgentTask(
        id="t",
        question="q",
        expected_titles=["Platform Engineer"],
        forbidden_titles=["Backend Engineer"],
        expected_numbers=[15750],
    )

    assert check_answer(task, "Platform Engineer pays 15,750 PLN on average.", TITLES) == []


def test_check_answer_lists_every_problem() -> None:
    task = AgentTask(
        id="t",
        question="q",
        expected_titles=["Platform Engineer", "LLM Engineer"],
        forbidden_titles=["Backend Engineer"],
        expected_numbers=[15750],
    )

    problems = check_answer(task, "llm engineer and Backend Engineer, about 16,000.", TITLES)

    assert problems == [
        "missing Platform Engineer",
        "unexpected Backend Engineer",
        "missing number 15750",
    ]


def test_check_answer_tolerates_rounding_but_not_a_different_value() -> None:
    task = AgentTask(id="t", question="q", expected_numbers=[15750])

    assert check_answer(task, "about 15,800 PLN", TITLES) == []
    assert check_answer(task, "15,875 PLN", TITLES) == ["missing number 15750"]


def test_check_answer_expects_refusal_without_offers() -> None:
    task = AgentTask(id="t", question="q", refuse=True)

    assert check_answer(task, "I can only help with job offers.", TITLES) == []
    assert check_answer(task, "Try LLM Engineer!", TITLES) == [
        "should refuse, but mentions LLM Engineer"
    ]


def test_agent_tasks_point_to_existing_offers() -> None:
    titles = {offer.title for offer in load_offers("offers.json")}

    for task in load_tasks(TASKS_PATH):
        assert set(task.expected_titles) | set(task.forbidden_titles) <= titles, task.id


def test_agent_task_ids_are_unique() -> None:
    ids = [task.id for task in load_tasks(TASKS_PATH)]

    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("task", load_tasks(TASKS_PATH), ids=lambda task: task.id)
def test_agent_task_checks_something(task: AgentTask) -> None:
    assert task.expected_titles or task.expected_numbers or task.refuse
