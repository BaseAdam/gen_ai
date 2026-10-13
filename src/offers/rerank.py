from google import genai
from google.genai import types
from pydantic import ValidationError

from offers.models import Relevance, ScoredOffer

RERANK_INSTRUCTION = (
    "You rate how well fragments of job ads answer the user's question. "
    "Score each fragment from 0 to 10 using this scale: "
    "0 = unrelated to the question; "
    "1-4 = related topic, but the fragment does not answer the question; "
    "5-6 = the fragment answers indirectly or partially, for example it mentions "
    "a broader category that likely includes what the user asks about; "
    "7-9 = the fragment answers the question, with minor gaps; "
    "10 = the fragment answers the question directly. "
    "Judge only by the content of the fragment. "
    "Return a score for every fragment, using its number as the index."
)
MIN_RELEVANCE = 5


def build_rerank_prompt(question: str, candidates: list[ScoredOffer]) -> str:
    blocks = [f"[{number}] {item.chunk}" for number, item in enumerate(candidates, start=1)]
    fragments = "\n\n".join(blocks)
    return f"QUESTION: {question}\n\nFRAGMENTS:\n{fragments}"


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
        item.model_copy(update={"score": scores.get(number, 0)})
        for number, item in numbered
        if scores.get(number, 0) >= min_relevance
    ]
    return ranked[:top_k]
