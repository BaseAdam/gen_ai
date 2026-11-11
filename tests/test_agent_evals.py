import pytest

from offers.agent import LIST_OFFERS, SALARY_STATS, SEARCH_JOB_ADS
from offers.agent_evals import AgentTask, check_answer, check_trajectory, load_tasks, numbers_in
from offers.io import load_offers

TASKS_PATH = "evals/agent_tasks.json"
TITLES = ["Platform Engineer", "LLM Engineer", "Backend Engineer"]
TOOL_NAMES = {LIST_OFFERS.name, SEARCH_JOB_ADS.name, SALARY_STATS.name}


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


def test_check_answer_reads_titles_from_offers_list_when_given() -> None:
    task = AgentTask(
        id="t",
        question="q",
        expected_titles=["Platform Engineer"],
        forbidden_titles=["Backend Engineer"],
    )
    answer = "Platform Engineer fits. Backend Engineer does not, it has no training budget."

    assert check_answer(task, answer, TITLES, offers=["platform engineer "]) == []
    assert check_answer(task, answer, TITLES, offers=["Backend Engineer"]) == [
        "missing Platform Engineer",
        "unexpected Backend Engineer",
    ]


def test_check_answer_flags_offers_that_do_not_exist() -> None:
    task = AgentTask(id="t", question="q", expected_titles=["LLM Engineer"])

    problems = check_answer(task, "", TITLES, offers=["LLM Engineer", "Senior LLM Engineer"])

    assert problems == ["unknown offer Senior LLM Engineer"]


def test_check_answer_refusal_with_offers_list() -> None:
    task = AgentTask(id="t", question="q", refuse=True)

    assert check_answer(task, "Sorry, I only know job offers.", TITLES, offers=[]) == []
    assert check_answer(task, "Sorry.", TITLES, offers=["LLM Engineer"]) == [
        "should refuse, but mentions LLM Engineer"
    ]


def test_check_answer_still_reads_numbers_from_text_with_offers_list() -> None:
    task = AgentTask(id="t", question="q", expected_numbers=[12250])

    assert check_answer(task, "The average is 12,250 PLN.", TITLES, offers=[]) == []
    assert check_answer(task, "The average is high.", TITLES, offers=[]) == ["missing number 12250"]


def test_check_trajectory_passes_a_path_within_the_rules() -> None:
    task = AgentTask(
        id="t",
        question="q",
        expected_tools=["search_job_ads", "salary_stats"],
        forbidden_tools=["list_offers"],
        max_steps=3,
    )

    assert check_trajectory(task, [["search_job_ads"], ["salary_stats"], []]) == []


def test_check_trajectory_lists_every_problem() -> None:
    task = AgentTask(
        id="t",
        question="q",
        expected_tools=["salary_stats"],
        forbidden_tools=["search_job_ads"],
        max_tool_calls=1,
        max_steps=2,
    )

    problems = check_trajectory(task, [["search_job_ads"], ["list_offers"], []])

    assert problems == [
        "never called salary_stats",
        "called search_job_ads",
        "2 tool calls, limit 1",
        "3 steps, limit 2",
    ]


def test_check_trajectory_counts_parallel_calls_as_one_step() -> None:
    task = AgentTask(id="t", question="q", max_tool_calls=2, max_steps=2)

    assert check_trajectory(task, [["salary_stats", "salary_stats"], []]) == []
    assert check_trajectory(task, [["salary_stats"], ["salary_stats"], []]) == ["3 steps, limit 2"]


def test_check_trajectory_limits_are_inclusive_and_zero_means_no_tools() -> None:
    task = AgentTask(id="t", question="q", max_tool_calls=0, max_steps=1)

    assert check_trajectory(task, [[]]) == []
    assert check_trajectory(task, [["list_offers"], []]) == [
        "1 tool calls, limit 0",
        "2 steps, limit 1",
    ]


def test_check_trajectory_without_rules_accepts_any_path() -> None:
    task = AgentTask(id="t", question="q")

    assert check_trajectory(task, [["list_offers"]] * 6 + [[]]) == []


def test_agent_tasks_use_existing_tools() -> None:
    for task in load_tasks(TASKS_PATH):
        assert set(task.expected_tools) | set(task.forbidden_tools) <= TOOL_NAMES, task.id
        assert not set(task.expected_tools) & set(task.forbidden_tools), task.id


@pytest.mark.parametrize("task", load_tasks(TASKS_PATH), ids=lambda task: task.id)
def test_agent_task_has_a_step_budget(task: AgentTask) -> None:
    assert task.max_steps is not None


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
