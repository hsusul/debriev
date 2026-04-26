"""Assertion ORM model."""

from uuid import UUID

from sqlalchemy import ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, IdMixin, TimestampMixin


class Assertion(Base, IdMixin, TimestampMixin):
    """Paragraph- or sentence-scoped factual assertion from a draft."""

    __tablename__ = "assertions"

    draft_id: Mapped[UUID] = mapped_column(
        ForeignKey("drafts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    paragraph_index: Mapped[int | None] = mapped_column(Integer)
    sentence_index: Mapped[int | None] = mapped_column(Integer)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_text: Mapped[str] = mapped_column(Text, nullable=False)
    extraction_strategy: Mapped[str | None] = mapped_column(String(32))
    extraction_version: Mapped[int | None] = mapped_column(Integer)
    extraction_snapshot: Mapped[dict[str, object] | None] = mapped_column(JSON)

    draft = relationship("Draft", back_populates="assertions")
    claim_units = relationship("ClaimUnit", back_populates="assertion", cascade="all, delete-orphan")
