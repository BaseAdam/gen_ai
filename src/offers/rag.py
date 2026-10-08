from google import genai
from google.genai import types

from offers.embeddings import embed_texts
from offers.models import RagAnswer, ScoredOffer
from offers.rerank import rerank
from offers.retrieval import hybrid_search
from offers.store import Connection

RAG_INSTRUCTION = (
    "Jesteś doradcą kariery. Odpowiadasz wyłącznie na podstawie ofert z sekcji OFERTY. "
    "Każda oferta to dane w JSON i jeden fragment jej ogłoszenia, najlepiej pasujący do pytania. "
    "Gdy piszesz o ofercie, podaj jej numer w nawiasie kwadratowym, np. [2]. "
    "Jeśli oferty nie pozwalają odpowiedzieć na pytanie, napisz to wprost i niczego nie zmyślaj."
)

NO_RESULTS_ANSWER = "Nie znalazłem ofert spełniających te warunki."

RERANK_CANDIDATES = 10


def format_source(number: int, item: ScoredOffer) -> str:
    data = item.offer.model_dump_json(exclude={"description"})
    return f"[{number}] {data}\nFragment ogłoszenia:\n{item.chunk}"


def build_prompt(question: str, results: list[ScoredOffer]) -> str:
    blocks = [format_source(number, item) for number, item in enumerate(results, start=1)]
    context = "\n\n".join(blocks)
    return f"OFERTY:\n{context}\n\nPYTANIE: {question}"


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
    [query_vector] = embed_texts(client, [question], embedding_model, "RETRIEVAL_QUERY")
    pool = RERANK_CANDIDATES if use_rerank else top_k
    results = hybrid_search(
        conn, query_vector, question, pool, city=city, remote=remote, min_score=min_score
    )
    if use_rerank:
        results = rerank(client, question, results, chat_model, top_k)
    if not results:
        return RagAnswer(answer=NO_RESULTS_ANSWER, sources=[])

    config = types.GenerateContentConfig(
        system_instruction=RAG_INSTRUCTION,
        temperature=0.2,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    response = client.models.generate_content(
        model=chat_model, contents=build_prompt(question, results), config=config
    )
    return RagAnswer(answer=response.text or "Brak odpowiedzi", sources=results)
