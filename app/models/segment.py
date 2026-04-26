"""Segment ORM model."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.provenance import anchor_sort_key, has_usable_anchor, normalize_anchor_metadata, render_anchor_text
from app.models.base import Base, IdMixin


class Segment(Base, IdMixin):
    """Anchored segment extracted from a source document."""

    __tablename__ = "segments"

    source_document_id: Mapped[UUID] = mapped_column(
        ForeignKey("source_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    page_start: Mapped[int | None] = mapped_column(Integer)
    line_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    line_end: Mapped[int | None] = mapped_column(Integer)
    anchor_metadata: Mapped[dict[str, object] | None] = mapped_column(JSON)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_text: Mapped[str] = mapped_column(Text, nullable=False)
    speaker: Mapped[str | None] = mapped_column(String(100))
    segment_type: Mapped[str] = mapped_column(String(50), default="TRANSCRIPT_BLOCK", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    source_document = relationship("SourceDocument", back_populates="segments")
    support_links = relationship("SupportLink", back_populates="segment")

    @property
    def normalized_anchor_metadata(self) -> dict[str, object] | None:
        return normalize_anchor_metadata(
            self.anchor_metadata,
            page_start=self.page_start,
            line_start=self.line_start,
            page_end=self.page_end,
            line_end=self.line_end,
        )

    @property
    def rendered_anchor(self) -> str:
        return render_anchor_text(
            self.anchor_metadata,
            page_start=self.page_start,
            line_start=self.line_start,
            page_end=self.page_end,
            line_end=self.line_end,
        )

    @property
    def has_usable_anchor(self) -> bool:
        return has_usable_anchor(
            self.anchor_metadata,
            page_start=self.page_start,
            line_start=self.line_start,
            page_end=self.page_end,
            line_end=self.line_end,
        )

    @property
    def normalized_anchor_sort_key(self) -> tuple[object, ...]:
        return anchor_sort_key(
            self.anchor_metadata,
            page_start=self.page_start,
            line_start=self.line_start,
            page_end=self.page_end,
            line_end=self.line_end,
            created_at=self.created_at,
            segment_id=self.id,
        )
