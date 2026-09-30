import pytest
from pydantic import ValidationError

from offers.config import Settings


def test_reads_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    settings = Settings(_env_file=None)
    assert settings.gemini_api_key.get_secret_value() == "test-key"


def test_missing_key_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValidationError, match="gemini_api_key"):
        Settings(_env_file=None)


def test_key_is_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert "test-key" not in repr(Settings(_env_file=None))
