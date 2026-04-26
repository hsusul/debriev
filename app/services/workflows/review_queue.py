"""Shared queue-state projection for draft review decisions."""

from dataclasses import dataclass, field
from uuid import UUID

from app.models import ClaimReviewDecision


@dataclass(slots=True)
class ReviewQueueProjection:
    total_flagged_claims: int
    resolved_flagged_claims: int
    remaining_flagged_claims: int
    next_claim_id: UUID | None
    active_claim_ids: list[UUID] = field(default_factory=list)
    resolved_claim_ids: list[UUID] = field(default_factory=list)


def build_latest_review_decision_index(
    decisions: list[ClaimReviewDecision],
) -> dict[UUID, ClaimReviewDecision]:
    latest_by_claim: dict[UUID, ClaimReviewDecision] = {}
    for decision in decisions:
        current = latest_by_claim.get(decision.claim_unit_id)
        if current is None or (decision.created_at, str(decision.id)) > (current.created_at, str(current.id)):
            latest_by_claim[decision.claim_unit_id] = decision
    return latest_by_claim


def project_review_queue(
    claim_ids_in_order: list[UUID],
    *,
    latest_decisions_by_claim: dict[UUID, ClaimReviewDecision],
    current_claim_id: UUID | None = None,
) -> ReviewQueueProjection:
    active_claim_ids: list[UUID] = []
    resolved_claim_ids: list[UUID] = []

    for claim_id in claim_ids_in_order:
        if latest_decisions_by_claim.get(claim_id) is None:
            active_claim_ids.append(claim_id)
        else:
            resolved_claim_ids.append(claim_id)

    next_claim_id = _next_claim_id(
        claim_ids_in_order=claim_ids_in_order,
        active_claim_ids=active_claim_ids,
        current_claim_id=current_claim_id,
    )
    return ReviewQueueProjection(
        total_flagged_claims=len(claim_ids_in_order),
        resolved_flagged_claims=len(resolved_claim_ids),
        remaining_flagged_claims=len(active_claim_ids),
        next_claim_id=next_claim_id,
        active_claim_ids=active_claim_ids,
        resolved_claim_ids=resolved_claim_ids,
    )


def _next_claim_id(
    *,
    claim_ids_in_order: list[UUID],
    active_claim_ids: list[UUID],
    current_claim_id: UUID | None,
) -> UUID | None:
    if not active_claim_ids:
        return None
    if current_claim_id is None:
        return active_claim_ids[0]

    try:
        current_index = claim_ids_in_order.index(current_claim_id)
    except ValueError:
        current_index = None

    if current_index is not None:
        for claim_id in claim_ids_in_order[current_index + 1 :]:
            if claim_id in active_claim_ids:
                return claim_id

    return active_claim_ids[0]
