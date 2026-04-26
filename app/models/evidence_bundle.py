"""Evidence bundle ORM model."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import Column, DateTime, ForeignKey, String, Table, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, IdMixin, TimestampMixin


evidence_bundle_source_documents = Table(
    "evidence_bundle_source_documents",
    Base.metadata,
    Column("evidence_bundle_id", Uuid(as_uuid=True), ForeignKey("evidence_bundles.id", ondelete="CASCADE"), primary_key=True),
    Column("source_document_id", Uuid(as_uuid=True), ForeignKey("source_documents.id", ondelete="CASCADE"), primary_key=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)


class EvidenceBundle(Base, IdMixin, TimestampMixin):
    """Explicit evidentiary source scope for a draft."""

    __tablename__ = "evidence_bundles"

    matter_id: Mapped[UUID] = mapped_column(
        ForeignKey("matters.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    matter = relationship("Matter", back_populates="evidence_bundles")
    drafts = relationship("Draft", back_populates="evidence_bundle")
    source_documents = relationship(
        "SourceDocument",
        secondary="evidence_bundle_source_documents",
        back_populates="evidence_bundles",
    )
