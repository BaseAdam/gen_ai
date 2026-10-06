import pytest

from offers.chunking import chunk_offer, split_sections
from offers.models import Offer

DESCRIPTION = """O firmie
Acme robi oprogramowanie.

Obowiązki
Piszesz kod.


Oferujemy
Umowę o pracę.
"""


def test_split_sections_splits_on_blank_lines() -> None:
    assert split_sections(DESCRIPTION) == [
        "O firmie\nAcme robi oprogramowanie.",
        "Obowiązki\nPiszesz kod.",
        "Oferujemy\nUmowę o pracę.",
    ]


@pytest.mark.parametrize("description", ["", "  \n\n  \n"])
def test_split_sections_of_blank_text_is_empty(description: str) -> None:
    assert split_sections(description) == []


def test_chunk_offer_prefixes_each_section_with_offer_header() -> None:
    offer = Offer(
        title="Dev",
        company="Acme",
        city="Gdańsk",
        remote=True,
        skills=["python"],
        description=DESCRIPTION,
    )
    header = "Dev w Acme, Gdańsk, zdalnie. Wymagania: python."

    assert chunk_offer(offer) == [
        f"{header}\nO firmie\nAcme robi oprogramowanie.",
        f"{header}\nObowiązki\nPiszesz kod.",
        f"{header}\nOferujemy\nUmowę o pracę.",
    ]


def test_chunk_offer_without_description_returns_header_only() -> None:
    offer = Offer(title="Dev", company="Acme", city="Gdańsk")

    assert chunk_offer(offer) == ["Dev w Acme, Gdańsk, stacjonarnie. Wymagania: brak danych."]
