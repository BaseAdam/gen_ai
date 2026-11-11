import argparse
import time

from google.genai import errors

from offers.agent import answer_with_agent
from offers.agent_evals import check_answer, load_tasks
from offers.assistant import answer_question
from offers.config import Settings
from offers.io import load_offers
from offers.llm import create_client
from offers.observability import configure_logging
from offers.rag import answer_with_rag
from offers.store import connect
from offers.tools import AgentDeps

TASKS_PATH = "evals/agent_tasks.json"
ALL_SYSTEMS = ["assistant", "rag", "agent"]
PAUSE_SECONDS = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agent tasks against the chosen systems.")
    parser.add_argument("systems", nargs="*", choices=ALL_SYSTEMS, help="default: all systems")
    parser.add_argument("--task", action="append", default=[], help="run only this task id")
    parser.add_argument("--verbose", action="store_true", help="show logs and full answers")
    return parser.parse_args()


def run_system(name: str, deps: AgentDeps, question: str) -> str:
    settings = deps.settings
    if name == "assistant":
        return answer_question(deps.client, question, deps.offers, settings.gemini_model)
    if name == "agent":
        return answer_with_agent(deps, question)
    result = answer_with_rag(
        deps.client,
        deps.conn,
        question,
        settings.gemini_model,
        settings.gemini_embedding_model,
        min_score=settings.rag_min_score,
        use_rerank=settings.rag_rerank,
    )
    return result.answer


if __name__ == "__main__":
    args = parse_args()
    systems = args.systems or ALL_SYSTEMS
    if args.verbose:
        configure_logging()
    settings = Settings()
    offers = load_offers("offers.json")
    titles = [offer.title for offer in offers]
    tasks = [task for task in load_tasks(TASKS_PATH) if not args.task or task.id in args.task]
    passed = dict.fromkeys(systems, 0)

    print(f"Tasks: {len(tasks)}, model: {settings.gemini_model}, systems: {', '.join(systems)}")
    with connect(settings.database_url) as conn:
        deps = AgentDeps(
            client=create_client(settings), conn=conn, settings=settings, offers=offers
        )
        for task in tasks:
            for name in systems:
                try:
                    answer = run_system(name, deps, task.question)
                    problems = check_answer(task, answer, titles)
                except (RuntimeError, errors.APIError) as error:
                    answer = ""
                    problems = [f"error: {error!r}"]
                if problems:
                    print(f"{task.id:<28}{name:<11}FAIL  {'; '.join(problems)}")
                else:
                    passed[name] += 1
                    print(f"{task.id:<28}{name:<11}pass")
                if args.verbose:
                    print(f"\n{answer}\n")
                elif problems:
                    print(f"{'':<39}answer: {answer[:160]!r}")
                time.sleep(PAUSE_SECONDS)

    print()
    for name in systems:
        print(f"{name:<11}{passed[name]}/{len(tasks)} tasks passed")
