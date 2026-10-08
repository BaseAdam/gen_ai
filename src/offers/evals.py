import json

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    id: str
    question: str
    relevant: list[str] = Field(description="Tytuły ofert, które powinny wrócić")
    city: str | None = None
    remote: bool | None = None


def load_cases(path: str) -> list[EvalCase]:
    with open(path, encoding="utf-8") as f:
        raw_cases = json.load(f)
    return [EvalCase.model_validate(raw) for raw in raw_cases]
