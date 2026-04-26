"""Re-extraction run repository."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import ReExtractionRunKind
from app.models import ReExtractionRun


class ReExtractionRunRepository:
    """Persistence helpers for draft-level re-extraction history."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        draft_id: UUID,
        *,
        run_kind: ReExtractionRunKind,
        requested_mode: str,
        extraction_version: int,
        total_assertions: int,
        ready_assertions: int,
        unchanged_assertions: int,
        applied_assertions: int,
        skipped_assertions: int,
        blocked_assertions: int,
        materially_changed_assertions: int,
        legacy_unversioned_assertions: int,
        replaced_assertions: int,
        metadata_only_assertions: int,
        snapshot_version: int,
        snapshot: dict[str, object],
    ) -> ReExtractionRun:
        run = ReExtractionRun(
            draft_id=draft_id,
            run_kind=run_kind,
            requested_mode=requested_mode,
            extraction_version=extraction_version,
            total_assertions=total_assertions,
            ready_assertions=ready_assertions,
            unchanged_assertions=unchanged_assertions,
            applied_assertions=applied_assertions,
            skipped_assertions=skipped_assertions,
            blocked_assertions=blocked_assertions,
            materially_changed_assertions=materially_changed_assertions,
            legacy_unversioned_assertions=legacy_unversioned_assertions,
            replaced_assertions=replaced_assertions,
            metadata_only_assertions=metadata_only_assertions,
            snapshot_version=snapshot_version,
            snapshot=snapshot,
        )
        self.session.add(run)
        self.session.flush()
        return run

    def list_by_draft(self, draft_id: UUID) -> list[ReExtractionRun]:
        stmt = (
            select(ReExtractionRun)
            .where(ReExtractionRun.draft_id == draft_id)
            .order_by(ReExtractionRun.created_at.desc())
        )
        return list(self.session.scalars(stmt))
