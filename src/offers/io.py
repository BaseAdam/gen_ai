import json
from collections.abc import Sequence

from pydantic import BaseModel

from offers.models import Offer


def load_models[M: BaseModel](path: str, model: type[M]) -> list[M]:
    with open(path, encoding="utf-8") as f:
        raw_items = json.load(f)
    return [model.model_validate(raw) for raw in raw_items]


def save_models(items: Sequence[BaseModel], path: str) -> None:
    raw_items = [item.model_dump(mode="json") for item in items]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw_items, f, ensure_ascii=False, indent=2)


def load_offers(path: str) -> list[Offer]:
    return load_models(path, Offer)


def save_offers(offers: list[Offer], path: str) -> None:
    save_models(offers, path)
