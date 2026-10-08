from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    gemini_api_key: SecretStr
    gemini_model: str
    gemini_timeout_seconds: float = Field(default=30, gt=0)
    gemini_embedding_model: str = "gemini-embedding-001"
    database_url: str = "postgresql://offers:offers@localhost:5432/offers"
    rag_min_score: float = Field(default=0.6, ge=0, le=1)
    rag_rerank: bool = True
