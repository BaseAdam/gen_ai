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


def salary_range(offers: list[Offer]) -> tuple[int, int] | None:
    values = [
        v
        for offer in offers
        if offer.salary_min is not None and offer.salary_max is not None
        for v in (offer.salary_min, offer.salary_max)
    ]
    return (min(values), max(values)) if values else None
