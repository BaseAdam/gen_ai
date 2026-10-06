from google import genai
from google.genai import types

from offers.embeddings import embed_texts
from offers.models import RagAnswer, ScoredOffer
from offers.store import Connection, search_offers

RAG_INSTRUCTION = (
    "Jesteś doradcą kariery. Odpowiadasz wyłącznie na podstawie ofert z sekcji OFERTY. "
    "Gdy piszesz o ofercie, podaj jej numer w nawiasie kwadratowym, np. [2]. "
    "Jeśli oferty nie pozwalają odpowiedzieć na pytanie, napisz to wprost i niczego nie zmyślaj."
)


def build_prompt(question: str, results: list[ScoredOffer]) -> str:
    lines = [
        f"[{number}] {item.offer.model_dump_json()}" for number, item in enumerate(results, start=1)
    ]
    context = "\n".join(lines)
    return f"OFERTY:\n{context}\n\nPYTANIE: {question}"


def answer_with_rag(
    client: genai.Client,
    conn: Connection,
    question: str,
    chat_model: str,
    embedding_model: str,
    top_k: int = 5,
) -> RagAnswer:
    [query_vector] = embed_texts(client, [question], embedding_model, "RETRIEVAL_QUERY")
    results = search_offers(conn, query_vector, top_k)

    config = types.GenerateContentConfig(
        system_instruction=RAG_INSTRUCTION,
        temperature=0.2,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    response = client.models.generate_content(
        model=chat_model, contents=build_prompt(question, results), config=config
    )
    return RagAnswer(answer=response.text or "Brak odpowiedzi", sources=results)
