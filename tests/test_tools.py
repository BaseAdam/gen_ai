import json
from typing import Any

from google.genai import types
from pydantic import SecretStr

from offers.config import Settings
from offers.embeddings import EMBEDDING_DIMENSIONS
from offers.models import Offer
from offers.store import Connection, replace_offers
from offers.tools import AgentDeps, list_offers, salary_stats, search_job_ads

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


def vec(*head: float) -> list[float]:
    return list(head) + [0.0] * (EMBEDDING_DIMENSIONS - len(head))


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


def make_deps(client: Any = None, conn: Any = None) -> AgentDeps:
    settings = Settings(
        gemini_api_key=SecretStr("test-key"),
        gemini_model="chat-model",
        rag_min_score=0.5,
        rag_rerank=True,
        _env_file=None,
    )
    return AgentDeps(client=client, conn=conn, settings=settings, offers=OFFERS)


def relevance(*scores: int) -> str:
    items = [{"index": number, "score": score} for number, score in enumerate(scores, start=1)]
    return json.dumps({"scores": items})


def test_list_offers_filters_and_leaves_out_descriptions() -> None:
    rows = list_offers(make_deps(), city="warsaw", remote=True)

    assert [row.title for row in rows] == ["A"]
    assert rows[0].skills == ["python"]
    assert "description" not in rows[0].model_dump()


def test_salary_stats_for_city() -> None:
    stats = salary_stats(make_deps(), city="Warsaw")

    assert (stats.offers_count, stats.average, stats.min, stats.max) == (2, 12500.0, 8000, 20000)


def test_salary_stats_for_given_titles_reports_gaps() -> None:
    stats = salary_stats(make_deps(), titles=["a", "C", "Ghost"])

    assert stats.offers_count == 2
    assert (stats.average, stats.min, stats.max) == (15000.0, 10000, 20000)
    assert stats.missing_salary_data == ["C"]
    assert stats.unknown_titles == ["Ghost"]


def test_salary_stats_combines_titles_with_filters() -> None:
    stats = salary_stats(make_deps(), titles=["A", "B", "C"], remote=True)

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

    hits = search_job_ads(make_deps(client, conn), "flexible hours")

    assert [(hit.title, hit.company, hit.fragment, hit.relevance) for hit in hits] == [
        ("B", "Y", "b", 9)
    ]
    assert (hits[0].salary_min, hits[0].salary_max) == (9000, 14000)
    assert len(client.models.generate_calls) == 1


def test_search_job_ads_returns_every_accepted_offer_best_first(conn: Connection) -> None:
    save_offers(conn)
    client = FakeClient()
    client.models.relevance = relevance(6, 8)

    hits = search_job_ads(make_deps(client, conn), "flexible hours")

    assert [hit.title for hit in hits] == ["C", "B"]


def test_search_job_ads_passes_filters_to_search(conn: Connection) -> None:
    save_offers(conn)
    client = FakeClient()
    client.models.relevance = relevance(9)

    hits = search_job_ads(make_deps(client, conn), "flexible hours", remote=True)

    assert [(hit.title, hit.remote) for hit in hits] == [("C", True)]
