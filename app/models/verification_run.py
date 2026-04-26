"""Verification run ORM model."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import SupportStatus
from app.models.base import Base, IdMixin

CURRENT_SUPPORT_SNAPSHOT_VERSION = 1


class VerificationRun(Base, IdMixin):
    """Immutable verification result for a claim unit."""

    __tablename__ = "verification_runs"

    claim_unit_id: Mapped[UUID] = mapped_column(
        ForeignKey("claim_units.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(100), nullable=False)
    deterministic_flags: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    reasoning_categories: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    verdict: Mapped[SupportStatus] = mapped_column(
        Enum(SupportStatus, name="verification_verdict_enum"),
        nullable=False,
    )
    reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    support_snapshot_version: Mapped[int | None] = mapped_column(Integer)
    support_snapshot: Mapped[dict[str, object] | None] = mapped_column(JSON)
    suggested_fix: Mapped[str | None] = mapped_column(Text)
    confidence_score: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    claim_unit = relationship("ClaimUnit", back_populates="verification_runs")
    claim_review_decisions = relationship(
        "ClaimReviewDecision",
        back_populates="verification_run",
    )
    user_decisions = relationship(
        "UserDecision",
        back_populates="verification_run",
        cascade="all, delete-orphan",
    )
