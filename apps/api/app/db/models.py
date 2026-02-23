from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, Column, String, Text

try:
    from sqlmodel import Field, SQLModel
except (
    ModuleNotFoundError
):  # pragma: no cover - allows extractor self-tests without db deps

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
    project_id: UUID | None = Field(
        default=None, foreign_key="project.project_id", index=True
    )
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
    context_text: str | None = Field(
        default=None, sa_column=Column(String(500), nullable=True)
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CitationVerification(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    input_hash: str = Field(index=True, unique=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    doc_id: str | None = None
    chunk_id: str | None = None
    raw_json: str = Field(sa_column=Column(Text, nullable=False))
    summary_status: str = Field(default="ambiguous")


class VerificationResult(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    doc_id: str = Field(index=True)
    input_hash: str = Field(index=True)
    citations_hash: str = Field(index=True)
    citations_json: str = Field(sa_column=Column(Text, nullable=False))
    findings_json: str = Field(sa_column=Column(Text, nullable=False))
    summary_json: str = Field(sa_column=Column(Text, nullable=False))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class VerificationJob(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True, index=True)
    doc_id: str = Field(index=True)
    status: str = Field(default="queued", sa_column=Column(String(16), nullable=False))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    error_text: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    result_id: int | None = Field(default=None, index=True)
