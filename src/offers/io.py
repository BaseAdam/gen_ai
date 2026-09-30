import json

from offers.models import Offer


def load_offers(path: str) -> list[Offer]:
    with open(path, encoding="utf-8") as f:
        raw_offers = json.load(f)
    return [Offer.model_validate(raw) for raw in raw_offers]


def save_offers(offers: list[Offer], path: str) -> None:
    dumped_offers = [offer.model_dump() for offer in offers]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dumped_offers, f, indent=2, ensure_ascii=False)
