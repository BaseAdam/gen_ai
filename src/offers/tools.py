from dataclasses import dataclass

from google import genai
from pydantic import BaseModel

from offers.analysis import avg_salary, filter_offers, salary_range
from offers.config import Settings
from offers.models import Offer
from offers.rag import RERANK_CANDIDATES, retrieve
from offers.store import Connection

MAX_LISTED_OFFERS = 25


@dataclass
class AgentDeps:
    client: genai.Client
    conn: Connection
    settings: Settings
    offers: list[Offer]


class OfferRow(BaseModel):
    title: str
    company: str
    city: str
    remote: bool
    salary_min: int | None
    salary_max: int | None
    skills: list[str]


class OfferList(BaseModel):
    total: int
    offers: list[OfferRow]


class SearchHit(BaseModel):
    title: str
    company: str
    city: str
    remote: bool
    salary_min: int | None
    salary_max: int | None
    fragment: str
    relevance: float


class SalaryStats(BaseModel):
    offers_count: int
    average: float | None
    min: int | None
    max: int | None
    missing_salary_data: list[str]
    unknown_titles: list[str]


def list_offers(
    deps: AgentDeps,
    city: str | None = None,
    remote: bool | None = None,
    skill: str | None = None,
) -> OfferList:
    found = filter_offers(deps.offers, city=city, remote=remote, skill=skill)
    rows = [OfferRow.model_validate(offer, from_attributes=True) for offer in found]
    return OfferList(total=len(rows), offers=rows[:MAX_LISTED_OFFERS])


def search_job_ads(
    deps: AgentDeps,
    query: str,
    city: str | None = None,
    remote: bool | None = None,
) -> list[SearchHit]:
    results = retrieve(
        deps.client,
        deps.conn,
        query,
        deps.settings.gemini_model,
        deps.settings.gemini_embedding_model,
        top_k=RERANK_CANDIDATES,
        city=city,
        remote=remote,
        min_score=deps.settings.rag_min_score,
        use_rerank=deps.settings.rag_rerank,
    )
    return [
        SearchHit(
            title=item.offer.title,
            company=item.offer.company,
            city=item.offer.city,
            remote=item.offer.remote,
            salary_min=item.offer.salary_min,
            salary_max=item.offer.salary_max,
            fragment=item.chunk,
            relevance=item.score,
        )
        for item in results
    ]


def salary_stats(
    deps: AgentDeps,
    titles: list[str] | None = None,
    city: str | None = None,
    remote: bool | None = None,
    skill: str | None = None,
) -> SalaryStats:
    selected = filter_offers(deps.offers, city=city, remote=remote, skill=skill)
    unknown_titles: list[str] = []
    if titles is not None:
        known = {offer.title.lower() for offer in deps.offers}
        unknown_titles = [title for title in titles if title.lower() not in known]
        wanted = {title.lower() for title in titles}
        selected = [offer for offer in selected if offer.title.lower() in wanted]

    salaries = salary_range(selected)
    return SalaryStats(
        offers_count=len(selected),
        average=avg_salary(selected),
        min=salaries[0] if salaries else None,
        max=salaries[1] if salaries else None,
        missing_salary_data=[
            offer.title
            for offer in selected
            if offer.salary_min is None or offer.salary_max is None
        ],
        unknown_titles=unknown_titles,
    )
