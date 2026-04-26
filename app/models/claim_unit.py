"""Claim unit ORM model."""

from uuid import UUID

from sqlalchemy import Enum, ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import ClaimType, SupportStatus
from app.models.base import Base, IdMixin, TimestampMixin


class ClaimUnit(Base, IdMixin, TimestampMixin):
    """Minimal claim unit extracted from an assertion."""

    __tablename__ = "claim_units"

    assertion_id: Mapped[UUID] = mapped_column(
        ForeignKey("assertions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_text: Mapped[str] = mapped_column(Text, nullable=False)
    claim_type: Mapped[ClaimType] = mapped_column(Enum(ClaimType, name="claim_type_enum"), nullable=False)
    sequence_order: Mapped[int] = mapped_column(Integer, nullable=False)
    support_status: Mapped[SupportStatus] = mapped_column(
        Enum(SupportStatus, name="support_status_enum"),
        default=SupportStatus.UNVERIFIED,
        nullable=False,
    )

    assertion = relationship("Assertion", back_populates="claim_units")
    review_decisions = relationship(
        "ClaimReviewDecision",
        back_populates="claim_unit",
        cascade="all, delete-orphan",
    )
    support_links = relationship("SupportLink", back_populates="claim_unit", cascade="all, delete-orphan")
    verification_runs = relationship(
        "VerificationRun",
        back_populates="claim_unit",
        cascade="all, delete-orphan",
    )
