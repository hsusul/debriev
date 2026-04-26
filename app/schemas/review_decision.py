"""Claim-level review decision schemas."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel

from app.core.enums import ClaimReviewAction, SupportStatus
from app.schemas.common import ORMModel


class ClaimReviewDecisionCreate(BaseModel):
    action: ClaimReviewAction
    note: str | None = None
    proposed_replacement_text: str | None = None


class ClaimReviewDecisionRead(ORMModel):
    id: UUID
    claim_unit_id: UUID
    draft_id: UUID
    verification_run_id: UUID | None
    action: ClaimReviewAction
    note: str | None
    proposed_replacement_text: str | None
    created_at: datetime


class ClaimReviewStateRead(BaseModel):
    claim_id: UUID
    draft_id: UUID
    review_status: Literal["reviewed"]
    latest_action: ClaimReviewAction
    decision_count: int
    latest_verification_run_id: UUID | None
    latest_verdict: SupportStatus | None
    removed_from_active_queue: bool


class DraftReviewQueueStateRead(BaseModel):
    draft_id: UUID
    total_flagged_claims: int
    resolved_flagged_claims: int
    remaining_flagged_claims: int
    next_claim_id: UUID | None


class ClaimReviewDecisionMutationRead(BaseModel):
    decision: ClaimReviewDecisionRead
    claim_review_state: ClaimReviewStateRead
    draft_queue: DraftReviewQueueStateRead
