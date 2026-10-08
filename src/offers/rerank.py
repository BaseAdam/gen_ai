from google import genai
from google.genai import types
from pydantic import ValidationError

from offers.models import Relevance, ScoredOffer

RERANK_INSTRUCTION = (
    "Oceniasz, jak dobrze fragmenty ogłoszeń o pracę odpowiadają na pytanie użytkownika. "
    "Każdemu fragmentowi daj ocenę od 0 do 10: 10 to bezpośrednia odpowiedź na pytanie, "
    "0 to brak związku. Oceniaj wyłącznie na podstawie treści fragmentu. "
    "Zwróć ocenę dla każdego fragmentu, używając jego numeru jako index."
)

MIN_RELEVANCE = 5


def build_rerank_prompt(question: str, candidates: list[ScoredOffer]) -> str:
    blocks = [f"[{number}] {item.chunk}" for number, item in enumerate(candidates, start=1)]
    fragments = "\n\n".join(blocks)
    return f"PYTANIE: {question}\n\nFRAGMENTY:\n{fragments}"


def rerank(
    client: genai.Client,
    question: str,
    candidates: list[ScoredOffer],
    model: str,
    top_k: int = 5,
    min_relevance: int = MIN_RELEVANCE,
) -> list[ScoredOffer]:
    if not candidates:
        return []

    config = types.GenerateContentConfig(
        system_instruction=RERANK_INSTRUCTION,
        temperature=0,
        response_mime_type="application/json",
        response_schema=Relevance,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    response = client.models.generate_content(
        model=model, contents=build_rerank_prompt(question, candidates), config=config
    )

    try:
        relevance = Relevance.model_validate_json(response.text or "")
    except ValidationError:
        return candidates[:top_k]

    scores = {item.index: item.score for item in relevance.scores}
    numbered = list(enumerate(candidates, start=1))
    numbered.sort(key=lambda pair: scores.get(pair[0], 0), reverse=True)

    ranked = [
        item.model_copy(update={"score": scores[number]})
        for number, item in numbered
        if scores.get(number, 0) >= min_relevance
    ]
    return ranked[:top_k]
