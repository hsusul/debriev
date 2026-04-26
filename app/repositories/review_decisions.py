"""Claim-level review decision repository."""

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import ClaimReviewAction
from app.models import ClaimReviewDecision


class ClaimReviewDecisionRepository:
    """Persistence helpers for immutable claim review decisions."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        *,
        claim_unit_id: UUID,
        draft_id: UUID,
        verification_run_id: UUID | None,
        action: ClaimReviewAction,
        note: str | None,
        proposed_replacement_text: str | None,
    ) -> ClaimReviewDecision:
        decision = ClaimReviewDecision(
            claim_unit_id=claim_unit_id,
            draft_id=draft_id,
            verification_run_id=verification_run_id,
            action=action,
            note=note,
            proposed_replacement_text=proposed_replacement_text,
            created_at=datetime.now(timezone.utc),
        )
        self.session.add(decision)
        self.session.flush()
        return decision

    def list_by_claim(self, claim_id: UUID) -> list[ClaimReviewDecision]:
        stmt = (
            select(ClaimReviewDecision)
            .where(ClaimReviewDecision.claim_unit_id == claim_id)
            .order_by(ClaimReviewDecision.created_at.desc())
        )
        return list(self.session.scalars(stmt))

    def list_by_draft(self, draft_id: UUID) -> list[ClaimReviewDecision]:
        stmt = (
            select(ClaimReviewDecision)
            .where(ClaimReviewDecision.draft_id == draft_id)
            .order_by(ClaimReviewDecision.created_at.desc())
        )
        return list(self.session.scalars(stmt))

    def latest_by_draft(self, draft_id: UUID) -> dict[UUID, ClaimReviewDecision]:
        latest_by_claim: dict[UUID, ClaimReviewDecision] = {}
        for decision in self.list_by_draft(draft_id):
            if decision.claim_unit_id not in latest_by_claim:
                latest_by_claim[decision.claim_unit_id] = decision
        return latest_by_claim
