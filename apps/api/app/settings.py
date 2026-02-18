from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    database_url: str = "postgresql+psycopg://debriev:debrievdev@postgres:5432/debriev"
    storage_dir: str = "/tmp/debriev_uploads"
    retrieval_db_path: str = "./data/retrieval.db"
    embed_provider: str = "stub"
    openai_api_key: str | None = None


settings = Settings()
