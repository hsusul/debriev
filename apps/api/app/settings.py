from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
        env_file=(".env", "apps/api/.env"),
        env_file_encoding="utf-8",
    )

    database_url: str = "sqlite:///./app.db"
    storage_dir: str = "/tmp/debriev_uploads"
    retrieval_db_path: str = "./data/retrieval.db"
    embed_provider: str = "stub"
    openai_api_key: str | None = None
    courtlistener_token: str | None = None


settings = Settings()
