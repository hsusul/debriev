from __future__ import annotations

from typing import Any

try:
    from sqlmodel import Session, SQLModel, create_engine
except ModuleNotFoundError:  # pragma: no cover - allows extractor self-tests without db deps
    class _DummyMetadata:
        def create_all(self, *args: Any, **kwargs: Any) -> None:
            return None

    class SQLModel:
        metadata = _DummyMetadata()

    class Session:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ModuleNotFoundError("sqlmodel is required for database-backed API routes")

    def create_engine(*args: Any, **kwargs: Any) -> None:  # type: ignore[no-redef]
        return None

from app.settings import settings


engine = create_engine(settings.database_url, echo=False)


def init_db() -> None:
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
