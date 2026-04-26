"""Verification run repository."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import VerificationRun


class VerificationRepository:
    """Persistence helpers for verification runs."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        claim_id: UUID,
        *,
        model_version: str,
        prompt_version: str,
        deterministic_flags: list[str],
        reasoning_categories: list[str],
        verdict: object,
        reasoning: str,
        support_snapshot_version: int | None,
        support_snapshot: dict[str, object] | None,
        suggested_fix: str | None,
        confidence_score: float | None,
    ) -> VerificationRun:
        run = VerificationRun(
            claim_unit_id=claim_id,
            model_version=model_version,
            prompt_version=prompt_version,
            deterministic_flags=deterministic_flags,
            reasoning_categories=reasoning_categories,
            verdict=verdict,
            reasoning=reasoning,
            support_snapshot_version=support_snapshot_version,
            support_snapshot=support_snapshot,
            suggested_fix=suggested_fix,
            confidence_score=confidence_score,
        )
        self.session.add(run)
        self.session.flush()
        return run

    def get(self, run_id: UUID) -> VerificationRun | None:
        return self.session.get(VerificationRun, run_id)

    def list_by_claim(self, claim_id: UUID) -> list[VerificationRun]:
        stmt = (
            select(VerificationRun)
            .where(VerificationRun.claim_unit_id == claim_id)
            .order_by(VerificationRun.created_at.desc())
        )
        return list(self.session.scalars(stmt))
