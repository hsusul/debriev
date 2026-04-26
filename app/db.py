"""Database engine and session helpers."""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


def build_engine(database_url: str | None = None, *, echo: bool | None = None) -> Engine:
    """Create a SQLAlchemy engine with sqlite-safe defaults for tests."""

    settings = get_settings()
    url = database_url or settings.database_url
    connect_args: dict[str, object] = {}

    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    return create_engine(
        url,
        echo=settings.sql_echo if echo is None else echo,
        future=True,
        connect_args=connect_args,
    )


settings = get_settings()
engine = build_engine(settings.database_url, echo=settings.sql_echo)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    """Yield a database session for request-scoped dependencies."""

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

