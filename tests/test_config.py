import pytest
from pydantic import ValidationError

from offers.config import Settings


def test_reads_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    settings = Settings(_env_file=None)
    assert settings.gemini_api_key.get_secret_value() == "test-key"
    assert settings.gemini_model == "test-model"


def test_missing_key_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    with pytest.raises(ValidationError, match="gemini_api_key"):
        Settings(_env_file=None)


def test_key_is_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    assert "test-key" not in repr(Settings(_env_file=None))


def test_timeout_default_and_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.delenv("GEMINI_TIMEOUT_SECONDS", raising=False)
    assert Settings(_env_file=None).gemini_timeout_seconds == 30

    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "5")
    assert Settings(_env_file=None).gemini_timeout_seconds == 5


def test_timeout_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValidationError, match="gemini_timeout_seconds"):
        Settings(_env_file=None)
