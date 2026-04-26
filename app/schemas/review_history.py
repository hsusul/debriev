"""Claim review history schemas."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel

from app.core.enums import ClaimGraphRelationshipType, ClaimReviewAction, StructuredReasoningCategory, SupportStatus
from app.schemas.review_decision import ClaimReviewDecisionRead
from app.schemas.verification import VerificationRunRead


class ClaimReviewHistoryChangeSummaryRead(BaseModel):
    latest_verdict: SupportStatus | None
    previous_verdict: SupportStatus | None
    verdict_changed: bool
    latest_confidence_score: float | None
    previous_confidence_score: float | None
    confidence_changed: bool
    latest_primary_anchor: str | None
    previous_primary_anchor: str | None
    primary_anchor_changed: bool
    latest_flags: list[str]
    previous_flags: list[str]
    flags_changed: bool
    latest_reasoning_categories: list[StructuredReasoningCategory]
    previous_reasoning_categories: list[StructuredReasoningCategory]
    reasoning_categories_changed: bool
    latest_support_assessment_count: int
    previous_support_assessment_count: int
    latest_excluded_link_count: int
    previous_excluded_link_count: int
    support_changed: bool
    changed_since_last_run: bool
    latest_decision_at: datetime | None
    latest_action: ClaimReviewAction | None


class ClaimReviewGraphRelationshipRead(BaseModel):
    relationship_type: ClaimGraphRelationshipType
    related_claim_id: UUID
    related_claim_text: str
    reason_code: str | None
    reason_text: str | None
    confidence_score: float | None


class ClaimReviewHistoryRead(BaseModel):
    claim_id: UUID
    draft_id: UUID
    claim_text: str
    assertion_context: str | None
    support_status: SupportStatus
    review_disposition: Literal["active", "resolved"]
    latest_decision: ClaimReviewDecisionRead | None
    decision_history: list[ClaimReviewDecisionRead]
    latest_verification: VerificationRunRead | None
    previous_verification: VerificationRunRead | None
    verification_runs: list[VerificationRunRead]
    reasoning_categories: list[StructuredReasoningCategory]
    contradiction_flags: list[str]
    claim_relationships: list[ClaimReviewGraphRelationshipRead]
    change_summary: ClaimReviewHistoryChangeSummaryRead
