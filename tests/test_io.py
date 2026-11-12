from pathlib import Path

import pytest
from pydantic import ValidationError

from offers.evals import EvalCase
from offers.io import load_models, load_offers, save_models, save_offers
from offers.models import Offer


def test_models_survive_a_round_trip(tmp_path: Path) -> None:
    path = str(tmp_path / "cases.json")
    cases = [
        EvalCase(id="a", question="Q?", relevant=["X"], city="Kraków"),
        EvalCase(id="b", question="R?", relevant=[]),
    ]

    save_models(cases, path)

    assert load_models(path, EvalCase) == cases


def test_saved_file_is_readable_utf8_with_indentation(tmp_path: Path) -> None:
    path = tmp_path / "offers.json"

    save_offers([Offer(title="Dev", company="Łódź Soft", city="Łódź")], str(path))

    text = path.read_text(encoding="utf-8")
    assert '"city": "Łódź"' in text
    assert text.startswith('[\n  {\n    "title": "Dev"')
    assert load_offers(str(path))[0].company == "Łódź Soft"


def test_load_models_rejects_invalid_items(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text('[{"id": "a"}]', encoding="utf-8")

    with pytest.raises(ValidationError):
        load_models(str(path), EvalCase)
