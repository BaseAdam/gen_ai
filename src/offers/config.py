from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

EVAL_TIMEOUT_SECONDS = 90.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    gemini_api_key: SecretStr
    gemini_model: str
    gemini_timeout_seconds: float = Field(default=30, gt=0)
    gemini_retry_attempts: int = Field(default=3, ge=1)
    gemini_embedding_model: str = "gemini-embedding-001"
    database_url: str = "postgresql://offers:offers@localhost:5432/offers"
    rag_min_score: float = Field(default=0.53, ge=0, le=1)
    rag_rerank: bool = True
    gemini_judge_model: str | None = None
    log_json: bool = False


def eval_settings() -> Settings:
    """Settings for batch evals: like the app's, but with a model timeout of at least 90 s.

    The API keeps the short timeout because a user is waiting; an eval would rather wait
    than lose a run to a slow response.
    """
    settings = Settings()
    timeout = max(settings.gemini_timeout_seconds, EVAL_TIMEOUT_SECONDS)
    return settings.model_copy(update={"gemini_timeout_seconds": timeout})
