from google import genai
from google.genai import types

from offers.models import RagAnswer, Verdict
from offers.rag import build_prompt

JUDGE_INSTRUCTION = (
    "You evaluate an answer written by a career assistant. "
    "The assistant could use only the offers in the OFFERS section. "
    "First write the reasoning: go through the claims in the answer one by one "
    "and check each of them against the offers. Then give three verdicts: "
    "grounded = every claim is supported by the offers, "
    "and a claim that goes beyond what an offer says is not supported; "
    "citations_correct = every number in square brackets points to an offer "
    "that supports the claim it is attached to; "
    "answers_question = the answer addresses what the user asked. "
    "Saying plainly that the offers do not cover something counts as a supported claim."
)


def build_judge_prompt(question: str, answer: RagAnswer) -> str:
    context = build_prompt(question, answer.sources)
    return f"{context}\n\nANSWER:\n{answer.answer}"


def judge_answer(client: genai.Client, question: str, answer: RagAnswer, model: str) -> Verdict:
    config = types.GenerateContentConfig(
        system_instruction=JUDGE_INSTRUCTION,
        temperature=0,
        response_mime_type="application/json",
        response_schema=Verdict,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    response = client.models.generate_content(
        model=model, contents=build_judge_prompt(question, answer), config=config
    )
    return Verdict.model_validate_json(response.text or "")
