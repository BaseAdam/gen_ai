import argparse
import time
from dataclasses import dataclass, field

import httpx
import pydantic_ai
from google.genai import errors
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

from offers.agent import answer_with_agent
from offers.agent_evals import AgentTask, check_answer, check_trajectory, load_tasks
from offers.assistant import answer_question
from offers.config import eval_settings
from offers.graph_agent import ask_graph
from offers.io import load_offers
from offers.llm import create_client
from offers.observability import RequestContext, configure_logging, request_context
from offers.pyai_agent import ask_agent
from offers.rag import answer_with_rag
from offers.store import connect
from offers.tools import AgentDeps

TASKS_PATH = "evals/agent_tasks.json"
ALL_SYSTEMS = ["assistant", "rag", "agent", "pyai", "graph"]
TOOL_SYSTEMS = {"agent", "pyai", "graph"}
INFRASTRUCTURE_ERRORS = (
    errors.APIError,
    httpx.TimeoutException,
    pydantic_ai.ModelAPIError,
    ChatGoogleGenerativeAIError,
)
PAUSE_SECONDS = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agent tasks against the chosen systems.")
    parser.add_argument("systems", nargs="*", choices=ALL_SYSTEMS, help="default: all systems")
    parser.add_argument("--task", action="append", default=[], help="run only this task id")
    parser.add_argument("--repeat", type=int, default=1, help="run every task this many times")
    parser.add_argument("--verbose", action="store_true", help="show logs and full answers")
    return parser.parse_args()


def run_system(name: str, deps: AgentDeps, question: str) -> tuple[str, list[str] | None]:
    settings = deps.settings
    if name == "assistant":
        return answer_question(deps.client, question, deps.offers, settings.gemini_model), None
    if name == "agent":
        return answer_with_agent(deps, question), None
    if name == "pyai":
        output = ask_agent(deps, question)
        return output.answer, output.offers
    if name == "graph":
        output = ask_graph(deps, question)
        return output.answer, output.offers
    result = answer_with_rag(
        deps.client,
        deps.conn,
        question,
        settings.gemini_model,
        settings.gemini_embedding_model,
        min_score=settings.rag_min_score,
        use_rerank=settings.rag_rerank,
    )
    return result.answer, None


@dataclass
class RunResult:
    answer: str
    offers: list[str] | None
    problems: list[str]
    path_problems: list[str] | None
    run: RequestContext
    quota_exhausted: bool = False


@dataclass
class SystemStats:
    runs: int = 0
    passed: int = 0
    paths_ok: int = 0
    paths_judged: int = 0
    tasks: int = 0
    tasks_always_passed: int = 0
    usage: RequestContext = field(default_factory=lambda: RequestContext(request_id="total"))

    def add(self, result: RunResult) -> None:
        self.runs += 1
        if not result.problems:
            self.passed += 1
        if result.path_problems is not None:
            self.paths_judged += 1
            if not result.path_problems:
                self.paths_ok += 1
        run, total = result.run, self.usage
        total.llm_calls += run.llm_calls
        total.steps += run.steps
        total.input_tokens += run.input_tokens
        total.output_tokens += run.output_tokens
        total.thinking_tokens += run.thinking_tokens


def daily_quota_exhausted(error: Exception) -> bool:
    # Gemini names the exhausted quota in the message,
    # e.g. GenerateRequestsPerDayPerProjectPerModel.
    return "PerDay" in str(error)


def run_once(
    name: str, deps: AgentDeps, task: AgentTask, titles: list[str], label: str
) -> RunResult:
    offers_chosen: list[str] | None = None
    interrupted = quota_exhausted = False
    with request_context(f"{label}/{name}") as run:
        try:
            answer, offers_chosen = run_system(name, deps, task.question)
            problems = check_answer(task, answer, titles, offers_chosen)
        except (RuntimeError, *INFRASTRUCTURE_ERRORS) as error:
            answer = ""
            problems = [f"error: {error!r}"]
            interrupted = isinstance(error, INFRASTRUCTURE_ERRORS)
            quota_exhausted = daily_quota_exhausted(error)
    path_problems = (
        check_trajectory(task, run.steps) if name in TOOL_SYSTEMS and not interrupted else None
    )
    return RunResult(answer, offers_chosen, problems, path_problems, run, quota_exhausted)


def verdict(problems: list[str] | None) -> str:
    if problems is None:
        return "-"
    return "FAIL" if problems else "pass"


def print_result(label: str, name: str, result: RunResult, verbose: bool) -> None:
    steps = result.run.steps
    tools = " > ".join("+".join(step) for step in steps if step) or "no tools"
    print(
        f"{label:<28}{name:<11}{verdict(result.problems):<8}{verdict(result.path_problems):<6}"
        f"{len(steps):<7}{tools}"
    )
    for problem in result.problems:
        print(f"{'':<4}answer: {problem}")
    for problem in result.path_problems or []:
        print(f"{'':<4}path: {problem}")
    if verbose:
        print(f"\n{result.answer}\n")
        if result.offers is not None:
            print(f"offers: {result.offers}\n")
    elif result.problems:
        print(f"{'':<4}answer was: {result.answer[:160]!r}")


def print_summary(totals: dict[str, SystemStats], repeat: int) -> None:
    print(
        f"{'system':<11}{'answers':>9}{f'pass^{repeat}':>9}{'paths':>9}"
        f"{'steps/run':>11}{'LLM calls/run':>15}{'tokens/run':>12}"
    )
    for name, stats in totals.items():
        if not stats.runs:
            print(f"{name:<11}{'no finished runs':>18}")
            continue
        usage = stats.usage
        tokens = usage.input_tokens + usage.output_tokens + usage.thinking_tokens
        paths = f"{stats.paths_ok}/{stats.paths_judged}" if name in TOOL_SYSTEMS else "-"
        print(
            f"{name:<11}{f'{stats.passed}/{stats.runs}':>9}"
            f"{f'{stats.tasks_always_passed}/{stats.tasks}':>9}{paths:>9}"
            f"{len(usage.steps) / stats.runs:>11.1f}{usage.llm_calls / stats.runs:>15.1f}"
            f"{tokens / stats.runs:>12,.0f}"
        )


def run_all(
    deps: AgentDeps,
    tasks: list[AgentTask],
    totals: dict[str, SystemStats],
    titles: list[str],
    args: argparse.Namespace,
) -> None:
    for task in tasks:
        for name, stats in totals.items():
            passed_before = stats.passed
            for attempt in range(1, args.repeat + 1):
                label = task.id if args.repeat == 1 else f"{task.id}#{attempt}"
                result = run_once(name, deps, task, titles, label)
                if result.quota_exhausted:
                    print(f"\n{label} {name}: daily Gemini quota exhausted, stopping.")
                    print("The summary covers the runs that finished.")
                    return
                stats.add(result)
                print_result(label, name, result, args.verbose)
                time.sleep(PAUSE_SECONDS)
            stats.tasks += 1
            if stats.passed - passed_before == args.repeat:
                stats.tasks_always_passed += 1


if __name__ == "__main__":
    pydantic_ai.BANNER_ENABLED = False
    args = parse_args()
    systems = args.systems or ALL_SYSTEMS
    if args.verbose:
        configure_logging()
    settings = eval_settings()
    offers = load_offers("offers.json")
    titles = [offer.title for offer in offers]
    tasks = [task for task in load_tasks(TASKS_PATH) if not args.task or task.id in args.task]
    totals = {name: SystemStats() for name in systems}

    print(
        f"Tasks: {len(tasks)} x {args.repeat}, model: {settings.gemini_model}, "
        f"systems: {', '.join(systems)}"
    )
    print(f"{'task':<28}{'system':<11}{'answer':<8}{'path':<6}{'steps':<7}tools")
    with connect(settings.database_url) as conn:
        deps = AgentDeps(
            client=create_client(settings), conn=conn, settings=settings, offers=offers
        )
        run_all(deps, tasks, totals, titles, args)

    print()
    print_summary(totals, args.repeat)
