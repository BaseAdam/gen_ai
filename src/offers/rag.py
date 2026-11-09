from google import genai
from google.genai import types

from offers.embeddings import embed_texts
from offers.models import RagAnswer, ScoredOffer
from offers.observability import log_usage, timed
from offers.rerank import rerank
from offers.retrieval import hybrid_search
from offers.store import Connection

RAG_INSTRUCTION = (
    "You are a career advisor. Answer only based on the offers in the OFFERS section. "
    "Each offer is JSON data plus the one fragment of its job ad that best matches the question. "
    "When you mention an offer, give its number in square brackets, e.g. [2]. "
    "If the offers do not answer the question, say so plainly and do not make anything up."
)

NO_RESULTS_ANSWER = "I couldn't find any offers matching these criteria."

RERANK_CANDIDATES = 10


def format_source(number: int, item: ScoredOffer) -> str:
    data = item.offer.model_dump_json(exclude={"description"})
    return f"[{number}] {data}\nJob ad fragment:\n{item.chunk}"


def build_prompt(question: str, results: list[ScoredOffer]) -> str:
    blocks = [format_source(number, item) for number, item in enumerate(results, start=1)]
    context = "\n\n".join(blocks)
    return f"OFFERS:\n{context}\n\nQUESTION: {question}"


def answer_with_rag(
    client: genai.Client,
    conn: Connection,
    question: str,
    chat_model: str,
    embedding_model: str,
    top_k: int = 5,
    city: str | None = None,
    remote: bool | None = None,
    min_score: float | None = None,
    use_rerank: bool = False,
) -> RagAnswer:
    with timed("rag.embed"):
        [query_vector] = embed_texts(client, [question], embedding_model, "RETRIEVAL_QUERY")
    pool = RERANK_CANDIDATES if use_rerank else top_k
    with timed("rag.search"):
        results = hybrid_search(
            conn, query_vector, question, pool, city=city, remote=remote, min_score=min_score
        )
    if use_rerank:
        with timed("rag.rerank"):
            results = rerank(client, question, results, chat_model, top_k)
    if not results:
        return RagAnswer(answer=NO_RESULTS_ANSWER, sources=[])

    config = types.GenerateContentConfig(
        system_instruction=RAG_INSTRUCTION,
        temperature=0.2,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    with timed("rag.generate"):
        response = client.models.generate_content(
            model=chat_model, contents=build_prompt(question, results), config=config
        )

    log_usage("rag.generate", response)
    return RagAnswer(answer=response.text or "No answer", sources=results)
