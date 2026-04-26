"""Support link ORM model."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import LinkType
from app.models.base import Base, IdMixin


class SupportLink(Base, IdMixin):
    """Link between a claim unit and a supporting record segment."""

    __tablename__ = "support_links"
    __table_args__ = (
        UniqueConstraint(
            "claim_unit_id",
            "segment_id",
            name="uq_support_links_claim_unit_segment",
        ),
    )

    claim_unit_id: Mapped[UUID] = mapped_column(
        ForeignKey("claim_units.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    segment_id: Mapped[UUID] = mapped_column(
        ForeignKey("segments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sequence_order: Mapped[int | None] = mapped_column(Integer)
    link_type: Mapped[LinkType] = mapped_column(Enum(LinkType, name="link_type_enum"), nullable=False)
    citation_text: Mapped[str | None] = mapped_column(Text)
    user_confirmed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    claim_unit = relationship("ClaimUnit", back_populates="support_links")
    segment = relationship("Segment", back_populates="support_links")
