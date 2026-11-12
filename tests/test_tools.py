from typing import Any

import pytest
from google.genai import types

from offers import tools
from offers.models import Offer
from offers.store import Connection, replace_offers
from offers.tools import list_offers, salary_stats, search_job_ads
from tests.helpers import make_deps, relevance, vec

OFFERS = [
    Offer(
        title="A",
        company="X",
        city="Warsaw",
        remote=True,
        salary_min=10000,
        salary_max=20000,
        skills=["python"],
        description="Long text.",
    ),
    Offer(title="B", company="Y", city="Warsaw", salary_min=8000, salary_max=12000),
    Offer(title="C", company="Z", city="Gdansk", remote=True, salary_max=11000),
]


class FakeModels:
    def __init__(self) -> None:
        self.relevance = '{"scores": []}'
        self.generate_calls: list[dict[str, Any]] = []

    def embed_content(self, **kwargs: Any) -> types.EmbedContentResponse:
        return types.EmbedContentResponse(embeddings=[types.ContentEmbedding(values=vec(1.0))])

    def generate_content(self, **kwargs: Any) -> types.GenerateContentResponse:
        self.generate_calls.append(kwargs)
        content = types.Content(role="model", parts=[types.Part(text=self.relevance)])
        return types.GenerateContentResponse(candidates=[types.Candidate(content=content)])


class FakeClient:
    def __init__(self) -> None:
        self.models = FakeModels()


def test_list_offers_filters_and_leaves_out_descriptions() -> None:
    result = list_offers(make_deps(OFFERS), city="warsaw", remote=True)

    assert result.total == 1
    assert [row.title for row in result.offers] == ["A"]
    assert result.offers[0].skills == ["python"]
    assert "description" not in result.offers[0].model_dump()


def test_list_offers_cuts_long_lists_but_keeps_the_total(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "MAX_LISTED_OFFERS", 2)

    result = list_offers(make_deps(OFFERS))

    assert result.total == 3
    assert [row.title for row in result.offers] == ["A", "B"]


def test_salary_stats_for_city() -> None:
    stats = salary_stats(make_deps(OFFERS), city="Warsaw")

    assert (stats.offers_count, stats.average, stats.min, stats.max) == (2, 12500.0, 8000, 20000)


def test_salary_stats_for_given_titles_reports_gaps() -> None:
    stats = salary_stats(make_deps(OFFERS), titles=["a", "C", "Ghost"])

    assert stats.offers_count == 2
    assert (stats.average, stats.min, stats.max) == (15000.0, 10000, 20000)
    assert stats.missing_salary_data == ["C"]
    assert stats.unknown_titles == ["Ghost"]


def test_salary_stats_combines_titles_with_filters() -> None:
    stats = salary_stats(make_deps(OFFERS), titles=["A", "B", "C"], remote=True)

    assert stats.offers_count == 2
    assert stats.average == 15000.0


def save_offers(conn: Connection) -> None:
    offers = [
        Offer(title="A", company="X", city="Warsaw"),
        Offer(title="B", company="Y", city="Krakow", salary_min=9000, salary_max=14000),
        Offer(title="C", company="Z", city="Gdansk", remote=True),
    ]
    vectors = [[vec(0.0, 1.0)], [vec(1.0, 0.0)], [vec(1.0, 1.0)]]
    replace_offers(conn, offers, [["a"], ["b"], ["c"]], vectors)


def test_search_job_ads_returns_offers_accepted_by_reranker(conn: Connection) -> None:
    save_offers(conn)
    client = FakeClient()
    client.models.relevance = relevance(9, 2)

    hits = search_job_ads(make_deps(OFFERS, client, conn, rag_min_score=0.5), "flexible hours")

    assert [(hit.title, hit.company, hit.fragment, hit.relevance) for hit in hits] == [
        ("B", "Y", "b", 9)
    ]
    assert (hits[0].salary_min, hits[0].salary_max) == (9000, 14000)
    assert len(client.models.generate_calls) == 1


def test_search_job_ads_returns_every_accepted_offer_best_first(conn: Connection) -> None:
    save_offers(conn)
    client = FakeClient()
    client.models.relevance = relevance(6, 8)

    hits = search_job_ads(make_deps(OFFERS, client, conn, rag_min_score=0.5), "flexible hours")

    assert [hit.title for hit in hits] == ["C", "B"]


def test_search_job_ads_passes_filters_to_search(conn: Connection) -> None:
    save_offers(conn)
    client = FakeClient()
    client.models.relevance = relevance(9)

    hits = search_job_ads(
        make_deps(OFFERS, client, conn, rag_min_score=0.5), "flexible hours", remote=True
    )

    assert [(hit.title, hit.remote) for hit in hits] == [("C", True)]
