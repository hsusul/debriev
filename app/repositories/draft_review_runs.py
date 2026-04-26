"""Draft review run repository."""

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import DraftReviewRunStatus
from app.models import DraftReviewRun


class DraftReviewRunRepository:
    """Persistence helpers for draft-level review execution history."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        draft_id: UUID,
        *,
        status: DraftReviewRunStatus,
        total_claims: int,
        total_flagged_claims: int,
        resolved_flagged_claims: int,
        remaining_flagged_claims: int,
        highest_severity_bucket: str | None,
        snapshot_version: int,
        snapshot: dict[str, object],
    ) -> DraftReviewRun:
        run = DraftReviewRun(
            draft_id=draft_id,
            status=status,
            total_claims=total_claims,
            total_flagged_claims=total_flagged_claims,
            resolved_flagged_claims=resolved_flagged_claims,
            remaining_flagged_claims=remaining_flagged_claims,
            highest_severity_bucket=highest_severity_bucket,
            snapshot_version=snapshot_version,
            snapshot=snapshot,
            created_at=datetime.now(timezone.utc),
        )
        self.session.add(run)
        self.session.flush()
        return run

    def list_by_draft(self, draft_id: UUID, *, limit: int | None = None) -> list[DraftReviewRun]:
        stmt = (
            select(DraftReviewRun)
            .where(DraftReviewRun.draft_id == draft_id)
            .order_by(DraftReviewRun.created_at.desc(), DraftReviewRun.id.desc())
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(self.session.scalars(stmt))
