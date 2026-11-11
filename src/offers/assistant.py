from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from offers.analysis import avg_salary, filter_offers, salary_range
from offers.models import Offer
from offers.tool_loop import run_tool_loop

SEARCH_OFFERS = types.FunctionDeclaration(
    name="search_offers",
    description=(
        "Searches job offers in the local database. All parameters are optional: "
        "omit the ones the user did not ask about."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City, e.g. Warsaw"},
            "remote": {"type": "boolean", "description": "true = remote offers only"},
            "skill": {"type": "string", "description": "A single technology, e.g. python"},
        },
    },
)


class SearchArgs(BaseModel):
    city: str | None = None
    remote: bool | None = None
    skill: str | None = None


def run_search_offers(args: dict[str, Any], offers: list[Offer]) -> list[dict[str, Any]]:
    parsed = SearchArgs.model_validate(args)
    found = filter_offers(offers, city=parsed.city, remote=parsed.remote, skill=parsed.skill)
    return [offer.model_dump() for offer in found]


SALARY_STATS = types.FunctionDeclaration(
    name="salary_stats",
    description=(
        "Computes salary statistics in PLN (average, minimum, maximum) for offers. "
        "Always use it when the question is about salaries. Do not calculate on your own."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City. Omit to include all cities."},
        },
    },
)


class SalaryArgs(BaseModel):
    city: str | None = None


def run_salary_stats(args: dict[str, Any], offers: list[Offer]) -> dict[str, Any]:
    parsed = SalaryArgs.model_validate(args)
    selected = filter_offers(offers, city=parsed.city)
    salaries = salary_range(selected)
    return {
        "offers_count": len(selected),
        "average": avg_salary(selected),
        "min": salaries[0] if salaries else None,
        "max": salaries[1] if salaries else None,
    }


ASSISTANT_INSTRUCTION = (
    "You are a job offer search assistant. Answer in English and keep it short. "
    "Take offer and salary data only from the tools; never guess or calculate on your own."
)


def assistant_config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=ASSISTANT_INSTRUCTION,
        temperature=0,
        tools=[types.Tool(function_declarations=[SEARCH_OFFERS, SALARY_STATS])],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


def run_tool(name: str | None, args: dict[str, Any], offers: list[Offer]) -> dict[str, Any]:
    try:
        if name == "search_offers":
            return {"result": run_search_offers(args, offers)}
        if name == "salary_stats":
            return {"result": run_salary_stats(args, offers)}
    except ValidationError as e:
        return {"error": f"Invalid arguments: {e}"}
    return {"error": f"Unknown tool: {name}"}


def answer_question(
    client: genai.Client, question: str, offers: list[Offer], model: str, max_steps: int = 5
) -> str:
    return run_tool_loop(
        client,
        model,
        question,
        assistant_config(),
        lambda name, args: run_tool(name, args, offers),
        max_steps,
        name="assistant",
    )
