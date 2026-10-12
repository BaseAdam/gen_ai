import pytest

from offers.chunking import chunk_offer, split_sections
from offers.models import Offer

DESCRIPTION = """About the company
Acme builds software.

Responsibilities
You write code.


What we offer
An employment contract.
"""


def test_split_sections_splits_on_blank_lines() -> None:
    assert split_sections(DESCRIPTION) == [
        "About the company\nAcme builds software.",
        "Responsibilities\nYou write code.",
        "What we offer\nAn employment contract.",
    ]


@pytest.mark.parametrize("description", ["", "  \n\n  \n"])
def test_split_sections_of_blank_text_is_empty(description: str) -> None:
    assert split_sections(description) == []


def test_chunk_offer_prefixes_each_section_with_offer_header() -> None:
    offer = Offer(
        title="Dev",
        company="Acme",
        city="Gdansk",
        remote=True,
        skills=["python"],
        description=DESCRIPTION,
    )
    header = "Dev at Acme, Gdansk, remote. Requirements: python."

    assert chunk_offer(offer) == [
        f"{header}\nAbout the company\nAcme builds software.",
        f"{header}\nResponsibilities\nYou write code.",
        f"{header}\nWhat we offer\nAn employment contract.",
    ]


def test_chunk_offer_without_description_returns_header_only() -> None:
    offer = Offer(title="Dev", company="Acme", city="Gdansk")

    assert chunk_offer(offer) == ["Dev at Acme, Gdansk, on-site. Requirements: not specified."]
