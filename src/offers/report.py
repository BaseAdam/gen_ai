from offers.analysis import avg_salary, group_by_city, top_skills
from offers.models import Offer


def print_report(offers: list[Offer]) -> None:
    print(f"Liczba ofert: {len(offers)}")
    for skill, count in top_skills(offers):
        print(f"  {skill}: {count}")
    for city, city_offers in group_by_city(offers).items():
        avg = avg_salary(city_offers)
        avg_str = f"{avg:.0f}" if avg is not None else "brak danych"
        print(f"{city}: {avg_str}")
    complete = [
        offer for offer in offers if offer.salary_min is not None and offer.salary_max is not None
    ]
    sorted_complete = sorted(complete, key=lambda offer: offer.salary_max or 0, reverse=True)
    for offer in sorted_complete[:3]:
        print(f"{offer.title:<30} {offer.company:<20} {offer.salary_min}-{offer.salary_max}")
