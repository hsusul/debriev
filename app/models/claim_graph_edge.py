"""Claim-graph relationship ORM model."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, Enum, Float, ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import ClaimGraphRelationshipType
from app.models.base import Base, IdMixin


class ClaimGraphEdge(Base, IdMixin):
    """Immutable claim relationship captured for a specific draft review run."""

    __tablename__ = "claim_graph_edges"
    __table_args__ = (
        UniqueConstraint(
            "draft_review_run_id",
            "source_claim_id",
            "target_claim_id",
            "relationship_type",
            name="uq_claim_graph_edge_run_source_target_type",
        ),
    )

    draft_id: Mapped[UUID] = mapped_column(
        ForeignKey("drafts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    draft_review_run_id: Mapped[UUID] = mapped_column(
        ForeignKey("draft_review_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_claim_id: Mapped[UUID] = mapped_column(
        ForeignKey("claim_units.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_claim_id: Mapped[UUID] = mapped_column(
        ForeignKey("claim_units.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    relationship_type: Mapped[ClaimGraphRelationshipType] = mapped_column(
        Enum(ClaimGraphRelationshipType, name="claim_graph_relationship_type_enum"),
        nullable=False,
    )
    reason_code: Mapped[str | None] = mapped_column(String(64))
    reason_text: Mapped[str | None] = mapped_column(Text)
    confidence_score: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    draft_review_run = relationship("DraftReviewRun", back_populates="claim_graph_edges")
