"""Claim-level review decision ORM model."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, Enum, ForeignKey, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import ClaimReviewAction
from app.models.base import Base, IdMixin


class ClaimReviewDecision(Base, IdMixin):
    """Immutable reviewer decision attached to a claim unit."""

    __tablename__ = "claim_review_decisions"

    claim_unit_id: Mapped[UUID] = mapped_column(
        ForeignKey("claim_units.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    draft_id: Mapped[UUID] = mapped_column(
        ForeignKey("drafts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    verification_run_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("verification_runs.id", ondelete="SET NULL"),
        index=True,
    )
    action: Mapped[ClaimReviewAction] = mapped_column(
        Enum(ClaimReviewAction, name="claim_review_action_enum"),
        nullable=False,
    )
    note: Mapped[str | None] = mapped_column(Text)
    proposed_replacement_text: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    claim_unit = relationship("ClaimUnit", back_populates="review_decisions")
    draft = relationship("Draft", back_populates="claim_review_decisions")
    verification_run = relationship("VerificationRun", back_populates="claim_review_decisions")
