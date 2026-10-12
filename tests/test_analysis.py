import pytest
from pydantic import ValidationError

from offers.analysis import avg_salary, filter_offers, salary_range
from offers.models import Offer


@pytest.fixture
def warsaw_remote() -> Offer:
    return Offer(
        title="AI Engineer",
        company="Lumen AI",
        city="Warsaw",
        salary_min=15000,
        salary_max=22000,
        remote=True,
        skills=["python", "openai"],
    )


@pytest.fixture
def krakow_onsite() -> Offer:
    return Offer(
        title="Backend Engineer",
        company="Neptune",
        city="Krakow",
        salary_min=12000,
        salary_max=18000,
        remote=False,
        skills=["python", "sql"],
    )


@pytest.fixture
def no_salary() -> Offer:
    return Offer(
        title="Data Analyst",
        company="Vistula",
        city="Wroclaw",
        remote=False,
        skills=["sql", "excel"],
    )


@pytest.fixture
def warsaw_remote_junior() -> Offer:
    return Offer(
        title="Junior Python Developer",
        company="Acme",
        city="Warsaw",
        salary_min=8000,
        salary_max=12000,
        remote=True,
        skills=["python", "docker"],
    )


@pytest.fixture
def all_offers(
    warsaw_remote: Offer,
    krakow_onsite: Offer,
    no_salary: Offer,
    warsaw_remote_junior: Offer,
) -> list[Offer]:
    return [warsaw_remote, krakow_onsite, no_salary, warsaw_remote_junior]


def test_avg_salary_returns_none_for_empty():
    assert avg_salary([]) is None


def test_avg_salary(warsaw_remote, krakow_onsite):
    assert avg_salary([warsaw_remote, krakow_onsite]) == 16750


def test_filtering_offers(warsaw_remote, krakow_onsite):
    assert filter_offers(offers=[warsaw_remote, krakow_onsite], remote=True) == [warsaw_remote]


@pytest.mark.parametrize(
    "skill_filter,expected_count", [("python", 3), ("excel", 1), ("ts", 0), ("Python", 3)]
)
def test_filter_offers_skills(all_offers, skill_filter, expected_count):
    assert len(filter_offers(all_offers, skill=skill_filter)) == expected_count


@pytest.mark.parametrize(
    "city_filter,expected_count",
    [
        ("Warsaw", 2),
        ("warsaw", 2),
        ("WARSAW", 2),
    ],
)
def test_filter_offers_city(all_offers, city_filter, expected_count):
    assert len(filter_offers(all_offers, city=city_filter)) == expected_count


def test_rejects_inverted_salary():
    with pytest.raises(ValidationError, match="Salary must be less than"):
        Offer(
            title="Junior Python Developer",
            company="Acme",
            city="Warsaw",
            salary_min=80000,
            salary_max=12000,
            remote=True,
            skills=["python", "docker"],
        )


def test_check_lowercase():
    sample = Offer(
        title="Junior Python Developer",
        company="Acme",
        city="Warsaw",
        salary_min=6000,
        salary_max=12000,
        remote=True,
        skills=["Python", "Aws"],
    )
    assert sample.skills == ["python", "aws"]


def test_salary_range(warsaw_remote):
    assert salary_range([warsaw_remote]) == (15000, 22000)


def test_empty_salary_range():
    assert salary_range([]) is None
