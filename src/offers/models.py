from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator


class Offer(BaseModel):
    title: str
    company: str
    city: str
    salary_min: int | None = None
    salary_max: int | None = None
    remote: bool = False
    skills: list[str] = Field(default=[], description="Required technologies")
    skills_nice_to_have: list[str] = Field(
        default=[], description="Technologies from the 'nice to have' section"
    )
    description: str = Field(default="", description="Full text of the job ad")

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
    chunk: str = ""
    offer_id: int = 0


class RagAnswer(BaseModel):
    answer: str
    sources: list[ScoredOffer]


class RelevanceScore(BaseModel):
    index: int
    score: int = Field(ge=0, le=10)


class Relevance(BaseModel):
    scores: list[RelevanceScore]


class Verdict(BaseModel):
    reasoning: str = Field(description="Claim-by-claim check, written before the verdicts")
    grounded: bool = Field(description="Every claim is supported by the offers")
    citations_correct: bool = Field(description="Every [n] supports the claim it is attached to")
    answers_question: bool = Field(description="The answer addresses what the user asked")
