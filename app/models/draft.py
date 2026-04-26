"""Draft ORM model."""

from uuid import UUID

from sqlalchemy import Enum, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import DraftMode
from app.models.base import Base, IdMixin, TimestampMixin


class Draft(Base, IdMixin, TimestampMixin):
    """Drafting artifact attached to a matter."""

    __tablename__ = "drafts"

    matter_id: Mapped[UUID] = mapped_column(
        ForeignKey("matters.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    evidence_bundle_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("evidence_bundles.id", ondelete="SET NULL"),
        index=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    mode: Mapped[DraftMode] = mapped_column(Enum(DraftMode, name="draft_mode_enum"), nullable=False)

    matter = relationship("Matter", back_populates="drafts")
    evidence_bundle = relationship("EvidenceBundle", back_populates="drafts")
    assertions = relationship("Assertion", back_populates="draft", cascade="all, delete-orphan")
    claim_review_decisions = relationship(
        "ClaimReviewDecision",
        back_populates="draft",
        cascade="all, delete-orphan",
    )
    reextraction_runs = relationship(
        "ReExtractionRun",
        back_populates="draft",
        cascade="all, delete-orphan",
    )
    draft_review_runs = relationship(
        "DraftReviewRun",
        back_populates="draft",
        cascade="all, delete-orphan",
    )
