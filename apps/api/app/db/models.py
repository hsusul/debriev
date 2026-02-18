from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, Column, String
try:
    from sqlmodel import Field, SQLModel
except ModuleNotFoundError:  # pragma: no cover - allows extractor self-tests without db deps
    class SQLModel:
        def __init_subclass__(cls, **kwargs: Any) -> None:
            super().__init_subclass__()

    def Field(
        default: Any = None,
        *,
        default_factory: Any = None,
        **kwargs: Any,
    ) -> Any:
        if default_factory is not None:
            return default_factory()
        return default


class Project(SQLModel, table=True):
    project_id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Document(SQLModel, table=True):
    doc_id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    project_id: UUID | None = Field(default=None, foreign_key="project.project_id", index=True)
    filename: str
    file_path: str
    stub_text: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Report(SQLModel, table=True):
    doc_id: UUID = Field(primary_key=True, foreign_key="document.doc_id")
    report_json: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Citation(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    doc_id: UUID = Field(foreign_key="document.doc_id", index=True)
    raw: str
    normalized: str | None = None
    start: int | None = None
    end: int | None = None
    context_text: str | None = Field(default=None, sa_column=Column(String(500), nullable=True))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
