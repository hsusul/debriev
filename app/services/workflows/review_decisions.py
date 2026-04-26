"""Claim-level review decision workflow for the draft review workbench."""

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.enums import ClaimReviewAction, SupportStatus
from app.core.exceptions import NotFoundError, ValidationError
from app.models import ClaimReviewDecision, VerificationRun
from app.repositories.claims import ClaimsRepository
from app.repositories.review_decisions import ClaimReviewDecisionRepository
from app.schemas.review_decision import ClaimReviewDecisionCreate
from app.services.workflows.review_queue import project_review_queue

FLAGGED_REVIEW_VERDICTS = frozenset(
    {
        SupportStatus.OVERSTATED,
        SupportStatus.AMBIGUOUS,
        SupportStatus.UNSUPPORTED,
        SupportStatus.UNVERIFIED,
    }
)


@dataclass(slots=True)
class ClaimReviewState:
    claim_id: UUID
    draft_id: UUID
    review_status: Literal["reviewed"]
    latest_action: ClaimReviewAction
    decision_count: int
    latest_verification_run_id: UUID | None
    latest_verdict: SupportStatus | None
    removed_from_active_queue: bool


@dataclass(slots=True)
class DraftReviewQueueState:
    draft_id: UUID
    total_flagged_claims: int
    resolved_flagged_claims: int
    remaining_flagged_claims: int
    next_claim_id: UUID | None


@dataclass(slots=True)
class ClaimReviewDecisionMutationResult:
    decision: ClaimReviewDecision
    claim_review_state: ClaimReviewState
    draft_queue: DraftReviewQueueState


@dataclass(slots=True)
class _ValidatedDecisionPayload:
    action: ClaimReviewAction
    note: str | None
    proposed_replacement_text: str | None


class ClaimReviewDecisionService:
    """Persist immutable claim review decisions and derive resulting queue state."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.claims = ClaimsRepository(session)
        self.review_decisions = ClaimReviewDecisionRepository(session)

    def record_decision(
        self,
        claim_id: UUID,
        payload: ClaimReviewDecisionCreate,
    ) -> ClaimReviewDecisionMutationResult:
        claim = self.claims.get(claim_id)
        if claim is None:
            raise NotFoundError("Claim unit not found.")

        draft_id = claim.assertion.draft_id
        validated = _validate_payload(payload)
        latest_run = _latest_verification_run(claim.verification_runs)

        decision = self.review_decisions.create(
            claim_unit_id=claim.id,
            draft_id=draft_id,
            verification_run_id=latest_run.id if latest_run is not None else None,
            action=validated.action,
            note=validated.note,
            proposed_replacement_text=validated.proposed_replacement_text,
        )
        decision_count = len(self.review_decisions.list_by_claim(claim.id))

        draft_claims = self.claims.list_by_draft(draft_id)
        flagged_claim_ids_in_order = [
            draft_claim.id
            for draft_claim in draft_claims
            if _current_persisted_verdict(draft_claim) in FLAGGED_REVIEW_VERDICTS
        ]
        latest_decisions_by_claim = self.review_decisions.latest_by_draft(draft_id)
        queue_projection = project_review_queue(
            flagged_claim_ids_in_order,
            latest_decisions_by_claim=latest_decisions_by_claim,
            current_claim_id=claim.id,
        )
        claim_review_state = ClaimReviewState(
            claim_id=claim.id,
            draft_id=draft_id,
            review_status="reviewed",
            latest_action=decision.action,
            decision_count=decision_count,
            latest_verification_run_id=latest_run.id if latest_run is not None else None,
            latest_verdict=latest_run.verdict if latest_run is not None else None,
            removed_from_active_queue=claim.id in queue_projection.resolved_claim_ids,
        )
        draft_queue = DraftReviewQueueState(
            draft_id=draft_id,
            total_flagged_claims=queue_projection.total_flagged_claims,
            resolved_flagged_claims=queue_projection.resolved_flagged_claims,
            remaining_flagged_claims=queue_projection.remaining_flagged_claims,
            next_claim_id=queue_projection.next_claim_id,
        )
        return ClaimReviewDecisionMutationResult(
            decision=decision,
            claim_review_state=claim_review_state,
            draft_queue=draft_queue,
        )


def _validate_payload(payload: ClaimReviewDecisionCreate) -> _ValidatedDecisionPayload:
    note = _clean_optional_text(payload.note)
    proposed_replacement_text = _clean_optional_text(payload.proposed_replacement_text)

    if payload.action == ClaimReviewAction.ACKNOWLEDGE_RISK:
        if proposed_replacement_text is not None:
            raise ValidationError("Acknowledge risk cannot include proposed replacement text.")
    elif payload.action == ClaimReviewAction.MARK_FOR_REVISION:
        if note is None:
            raise ValidationError("Mark for revision requires a note.")
        if proposed_replacement_text is not None:
            raise ValidationError("Mark for revision cannot include proposed replacement text.")
    elif payload.action == ClaimReviewAction.RESOLVE_WITH_EDIT and proposed_replacement_text is None:
        raise ValidationError("Resolve with edit requires proposed replacement text.")

    return _ValidatedDecisionPayload(
        action=payload.action,
        note=note,
        proposed_replacement_text=proposed_replacement_text,
    )


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _latest_verification_run(runs: list[VerificationRun]) -> VerificationRun | None:
    if not runs:
        return None
    return max(runs, key=lambda run: (run.created_at, str(run.id)))


def _current_persisted_verdict(claim: object) -> SupportStatus:
    latest_run = _latest_verification_run(claim.verification_runs)
    if latest_run is not None:
        return latest_run.verdict
    return claim.support_status
