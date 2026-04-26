"""Draft review execution run ORM model."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import DraftReviewRunStatus
from app.models.base import Base, IdMixin

CURRENT_DRAFT_REVIEW_RUN_SNAPSHOT_VERSION = 2


class DraftReviewRun(Base, IdMixin):
    """Immutable persisted history for fresh draft review executions."""

    __tablename__ = "draft_review_runs"

    draft_id: Mapped[UUID] = mapped_column(
        ForeignKey("drafts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[DraftReviewRunStatus] = mapped_column(
        Enum(DraftReviewRunStatus, name="draft_review_run_status_enum"),
        nullable=False,
    )
    total_claims: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_flagged_claims: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    resolved_flagged_claims: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    remaining_flagged_claims: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    highest_severity_bucket: Mapped[str | None] = mapped_column(String(32))
    snapshot_version: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    draft = relationship("Draft", back_populates="draft_review_runs")
    claim_graph_edges = relationship(
        "ClaimGraphEdge",
        back_populates="draft_review_run",
        cascade="all, delete-orphan",
    )
