from typing import Self

from pydantic import BaseModel, field_validator, model_validator


class Offer(BaseModel):
    title: str
    company: str
    city: str
    salary_min: int | None = None
    salary_max: int | None = None
    remote: bool = False
    skills: list[str] = []

    @model_validator(mode="after")
    def check_salary_range(self) -> Self:
        if self.salary_min is not None and self.salary_max is not None:
            if self.salary_min > self.salary_max:
                raise ValueError("Salary must be less than or equal to max")
        return self

    @field_validator("skills")
    @classmethod
    def normalize_skills(cls, value: list[str]) -> list[str]:
        return [skill.lower() for skill in value]

    def __str__(self) -> str:
        return (
            f"{self.title} @ {self.company} ({self.city}) {self.salary_min}-{self.salary_max} PLN"
        )
