"""Shared API dependencies."""

from collections.abc import Generator

from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import get_db


def get_db_session() -> Generator[Session, None, None]:
    """Provide a SQLAlchemy session."""

    yield from get_db()


def get_app_settings() -> Settings:
    """Provide application settings."""

    return get_settings()

