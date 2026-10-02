from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from offers.analysis import avg_salary, filter_offers, salary_range
from offers.models import Offer

SEARCH_OFFERS = types.FunctionDeclaration(
    name="search_offers",
    description=(
        "Wyszukuje oferty pracy w lokalnej bazie. Wszystkie parametry są opcjonalne: "
        "pomiń te, o które użytkownik nie pytał."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "Miasto, np. Warszawa"},
            "remote": {"type": "boolean", "description": "true = tylko oferty zdalne"},
            "skill": {"type": "string", "description": "Jedna technologia, np. python"},
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
        "Liczy statystyki widełek w PLN (średnia, minimum, maksimum) dla ofert. "
        "Używaj zawsze, gdy pytanie dotyczy zarobków. Nie licz samodzielnie."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "Miasto. Pomiń, żeby objąć wszystkie."},
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
    "Jesteś asystentem do wyszukiwania ofert pracy. Odpowiadaj po polsku i krótko. "
    "Dane o ofertach i zarobkach bierz wyłącznie z narzędzi, niczego nie zgaduj i nie licz sam."
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
        return {"error": f"Błędne argumenty: {e}"}
    return {"error": f"Nieznane narzędzie: {name}"}


def answer_question(
    client: genai.Client, question: str, offers: list[Offer], model: str, max_steps: int = 5
) -> str:
    contents: list[types.Content] = [types.Content(role="user", parts=[types.Part(text=question)])]

    for _ in range(max_steps):
        response = client.models.generate_content(
            model=model, contents=contents, config=assistant_config()
        )
        calls = response.function_calls
        if not calls:
            return response.text or "Brak odpowiedzi"

        if response.candidates and response.candidates[0].content:
            contents.append(response.candidates[0].content)

        for call in calls:
            result = run_tool(call.name, call.args or {}, offers)
            part = types.Part.from_function_response(name=call.name or "unknown", response=result)
            contents.append(types.Content(role="user", parts=[part]))

    raise RuntimeError("Model nie skończył w dozwolonej liczbie kroków")
