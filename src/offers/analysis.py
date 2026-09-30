from collections import Counter

from offers.models import Offer


def matches(offer: Offer, city: str | None, remote: bool | None, skill: str | None) -> bool:
    return (
        (city is None or offer.city.lower() == city.lower())
        and (remote is None or offer.remote == remote)
        and (skill is None or skill.lower() in offer.skills)
    )


def filter_offers(
    offers: list[Offer],
    city: str | None = None,
    remote: bool | None = None,
    skill: str | None = None,
) -> list[Offer]:
    return [offer for offer in offers if matches(offer, city, remote, skill)]


def avg_salary(offers: list[Offer]) -> float | None:
    per_offer_avgs = [
        (offer.salary_min + offer.salary_max) / 2
        for offer in offers
        if offer.salary_min is not None and offer.salary_max is not None
    ]
    return sum(per_offer_avgs) / len(per_offer_avgs) if per_offer_avgs else None


def top_skills(offers: list[Offer], n: int = 5) -> list[tuple[str, int]]:
    all_skills = [skill for offer in offers for skill in offer.skills]
    return most_common(all_skills, n)


def most_common[T](items: list[T], n: int = 5) -> list[tuple[T, int]]:
    counter = Counter(items)
    return counter.most_common(n)


def group_by_city(offers: list[Offer]) -> dict[str, list[Offer]]:
    groups: dict[str, list[Offer]] = {}
    for offer in offers:
        groups.setdefault(offer.city, []).append(offer)
    return groups


def salary_range(offers: list[Offer]) -> tuple[int, int] | None:
    values = [
        v
        for offer in offers
        if offer.salary_min is not None and offer.salary_max is not None
        for v in (offer.salary_min, offer.salary_max)
    ]
    return (min(values), max(values)) if values else None
