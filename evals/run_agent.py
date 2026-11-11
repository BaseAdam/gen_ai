import sys
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
SYSTEMS = [arg for arg in sys.argv[1:] if arg in ALL_SYSTEMS] or ALL_SYSTEMS
PAUSE_SECONDS = 15


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
    if "--verbose" in sys.argv:
        configure_logging()
    settings = Settings()
    offers = load_offers("offers.json")
    titles = [offer.title for offer in offers]
    tasks = load_tasks(TASKS_PATH)
    passed = dict.fromkeys(SYSTEMS, 0)

    print(f"Tasks: {TASKS_PATH}, model: {settings.gemini_model}, systems: {', '.join(SYSTEMS)}")
    with connect(settings.database_url) as conn:
        deps = AgentDeps(
            client=create_client(settings), conn=conn, settings=settings, offers=offers
        )
        for task in tasks:
            for name in SYSTEMS:
                try:
                    answer = run_system(name, deps, task.question)
                    problems = check_answer(task, answer, titles)
                except (RuntimeError, errors.APIError) as error:
                    answer = ""
                    problems = [f"error: {error!r}"]
                if problems:
                    print(f"{task.id:<28}{name:<11}FAIL  {'; '.join(problems)}")
                    print(f"{'':<39}answer: {answer[:160]!r}")
                else:
                    passed[name] += 1
                    print(f"{task.id:<28}{name:<11}pass")
                time.sleep(PAUSE_SECONDS)

    print()
    for name in SYSTEMS:
        print(f"{name:<11}{passed[name]}/{len(tasks)} tasks passed")
