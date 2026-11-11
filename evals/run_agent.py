import argparse
import time

import httpx
import pydantic_ai
from google.genai import errors

from offers.agent import answer_with_agent
from offers.agent_evals import check_answer, check_trajectory, load_tasks
from offers.assistant import answer_question
from offers.config import Settings
from offers.io import load_offers
from offers.llm import create_client
from offers.observability import RequestContext, configure_logging, request_context
from offers.pyai_agent import ask_agent
from offers.rag import answer_with_rag
from offers.store import connect
from offers.tools import AgentDeps

TASKS_PATH = "evals/agent_tasks.json"
ALL_SYSTEMS = ["assistant", "rag", "agent", "pyai"]
TOOL_SYSTEMS = {"agent", "pyai"}
INFRASTRUCTURE_ERRORS = (errors.APIError, httpx.TimeoutException, pydantic_ai.ModelAPIError)
PAUSE_SECONDS = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agent tasks against the chosen systems.")
    parser.add_argument("systems", nargs="*", choices=ALL_SYSTEMS, help="default: all systems")
    parser.add_argument("--task", action="append", default=[], help="run only this task id")
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


def add_usage(total: RequestContext, run: RequestContext) -> None:
    total.llm_calls += run.llm_calls
    total.steps += run.steps
    total.input_tokens += run.input_tokens
    total.output_tokens += run.output_tokens
    total.thinking_tokens += run.thinking_tokens


def verdict(problems: list[str] | None) -> str:
    if problems is None:
        return "-"
    return "FAIL" if problems else "pass"


def print_summary(
    systems: list[str],
    passed: dict[str, int],
    paths_ok: dict[str, int],
    paths_judged: dict[str, int],
    usage: dict[str, RequestContext],
    tasks: int,
) -> None:
    print(
        f"{'system':<11}{'answers':>8}{'paths':>8}"
        f"{'steps/task':>12}{'LLM calls/task':>16}{'tokens/task':>13}"
    )
    for name in systems:
        total = usage[name]
        tokens = total.input_tokens + total.output_tokens + total.thinking_tokens
        paths = f"{paths_ok[name]}/{paths_judged[name]}" if name in TOOL_SYSTEMS else "-"
        print(
            f"{name:<11}{f'{passed[name]}/{tasks}':>8}{paths:>8}{len(total.steps) / tasks:>12.1f}"
            f"{total.llm_calls / tasks:>16.1f}{tokens / tasks:>13,.0f}"
        )


if __name__ == "__main__":
    pydantic_ai.BANNER_ENABLED = False
    args = parse_args()
    systems = args.systems or ALL_SYSTEMS
    if args.verbose:
        configure_logging()
    settings = Settings()
    offers = load_offers("offers.json")
    titles = [offer.title for offer in offers]
    tasks = [task for task in load_tasks(TASKS_PATH) if not args.task or task.id in args.task]
    passed = dict.fromkeys(systems, 0)
    paths_ok = dict.fromkeys(systems, 0)
    paths_judged = dict.fromkeys(systems, 0)
    usage = {name: RequestContext(request_id=name) for name in systems}

    print(f"Tasks: {len(tasks)}, model: {settings.gemini_model}, systems: {', '.join(systems)}")
    print(f"{'task':<28}{'system':<11}{'answer':<8}{'path':<6}{'steps':<7}tools")
    with connect(settings.database_url) as conn:
        deps = AgentDeps(
            client=create_client(settings), conn=conn, settings=settings, offers=offers
        )
        for task in tasks:
            for name in systems:
                offers_chosen: list[str] | None = None
                interrupted = False
                with request_context(f"{task.id}/{name}") as run:
                    try:
                        answer, offers_chosen = run_system(name, deps, task.question)
                        problems = check_answer(task, answer, titles, offers_chosen)
                    except (RuntimeError, *INFRASTRUCTURE_ERRORS) as error:
                        answer = ""
                        problems = [f"error: {error!r}"]
                        interrupted = isinstance(error, INFRASTRUCTURE_ERRORS)
                path_problems = (
                    check_trajectory(task, run.steps)
                    if name in TOOL_SYSTEMS and not interrupted
                    else None
                )
                add_usage(usage[name], run)
                passed[name] += not problems
                paths_ok[name] += path_problems == []
                paths_judged[name] += path_problems is not None

                tools = " > ".join("+".join(step) for step in run.steps if step) or "no tools"
                print(
                    f"{task.id:<28}{name:<11}{verdict(problems):<8}{verdict(path_problems):<6}"
                    f"{len(run.steps):<7}{tools}"
                )
                for problem in problems:
                    print(f"{'':<4}answer: {problem}")
                for problem in path_problems or []:
                    print(f"{'':<4}path: {problem}")
                if args.verbose:
                    print(f"\n{answer}\n")
                    if offers_chosen is not None:
                        print(f"offers: {offers_chosen}\n")
                elif problems:
                    print(f"{'':<4}answer was: {answer[:160]!r}")
                time.sleep(PAUSE_SECONDS)

    print()
    print_summary(systems, passed, paths_ok, paths_judged, usage, len(tasks))
