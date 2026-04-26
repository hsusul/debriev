"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime settings for the Debriev API."""

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Debriev API"
    app_env: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    database_url: str = "postgresql+psycopg://debriev:debriev@localhost:5432/debriev"
    sql_echo: bool = False

    llm_provider: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    verification_model_version: str = "heuristic-v1"
    prompt_version: str = "v1"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings object."""

    return Settings()

