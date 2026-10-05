from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator


class Offer(BaseModel):
    title: str
    company: str
    city: str
    salary_min: int | None = None
    salary_max: int | None = None
    remote: bool = False
    skills: list[str] = Field(default=[], description="Wymagane technologie / Requirement tech")
    skills_nice_to_have: list[str] = Field(
        default=[], description="Technologie z sekcji 'mile widziane' / nice to have"
    )

    @model_validator(mode="after")
    def check_salary_range(self) -> Self:
        if self.salary_min is not None and self.salary_max is not None:
            if self.salary_min > self.salary_max:
                raise ValueError("Salary must be less than or equal to max")
        return self

    @field_validator("skills", "skills_nice_to_have")
    @classmethod
    def normalize_skills(cls, value: list[str]) -> list[str]:
        return [skill.lower() for skill in value]

    def __str__(self) -> str:
        return (
            f"{self.title} @ {self.company} ({self.city}) {self.salary_min}-{self.salary_max} PLN"
        )


class OfferSummary(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int


class ScoredOffer(BaseModel):
    offer: Offer
    score: float
