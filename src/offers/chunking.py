import re

from offers.embeddings import offer_to_text
from offers.models import Offer


def split_sections(description: str) -> list[str]:
    parts = re.split(r"\n\s*\n", description)
    return [part.strip() for part in parts if part.strip()]


def chunk_offer(offer: Offer) -> list[str]:
    header = offer_to_text(offer)
    sections = split_sections(offer.description)
    if not sections:
        return [header]
    return [f"{header}\n{section}" for section in sections]
