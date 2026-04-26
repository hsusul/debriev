"""Re-extraction migration run ORM model."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import ReExtractionRunKind
from app.models.base import Base, IdMixin

CURRENT_REEXTRACTION_RUN_SNAPSHOT_VERSION = 1


class ReExtractionRun(Base, IdMixin):
    """Immutable persisted history for draft-level re-extraction preview/apply runs."""

    __tablename__ = "reextraction_runs"

    draft_id: Mapped[UUID] = mapped_column(
        ForeignKey("drafts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    run_kind: Mapped[ReExtractionRunKind] = mapped_column(
        Enum(ReExtractionRunKind, name="reextraction_run_kind_enum"),
        nullable=False,
    )
    requested_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    extraction_version: Mapped[int] = mapped_column(Integer, nullable=False)
    total_assertions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ready_assertions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    unchanged_assertions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    applied_assertions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    skipped_assertions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    blocked_assertions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    materially_changed_assertions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    legacy_unversioned_assertions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    replaced_assertions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metadata_only_assertions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    snapshot_version: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    draft = relationship("Draft", back_populates="reextraction_runs")
