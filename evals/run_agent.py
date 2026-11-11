import time

from google import genai
from google.genai import errors

from offers.agent_evals import check_answer, load_tasks
from offers.assistant import answer_question
from offers.config import Settings
from offers.io import load_offers
from offers.llm import create_client
from offers.models import Offer
from offers.rag import answer_with_rag
from offers.store import Connection, connect

TASKS_PATH = "evals/agent_tasks.json"
SYSTEMS = ["assistant", "rag"]
PAUSE_SECONDS = 10


def run_system(
    name: str,
    client: genai.Client,
    conn: Connection,
    settings: Settings,
    offers: list[Offer],
    question: str,
) -> str:
    if name == "assistant":
        return answer_question(client, question, offers, settings.gemini_model)
    result = answer_with_rag(
        client,
        conn,
        question,
        settings.gemini_model,
        settings.gemini_embedding_model,
        min_score=settings.rag_min_score,
        use_rerank=settings.rag_rerank,
    )
    return result.answer


if __name__ == "__main__":
    settings = Settings()
    client = create_client(settings)
    offers = load_offers("offers.json")
    titles = [offer.title for offer in offers]
    tasks = load_tasks(TASKS_PATH)
    passed = dict.fromkeys(SYSTEMS, 0)

    print(f"Tasks: {TASKS_PATH}, model: {settings.gemini_model}")
    with connect(settings.database_url) as conn:
        for task in tasks:
            for name in SYSTEMS:
                try:
                    answer = run_system(name, client, conn, settings, offers, task.question)
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
