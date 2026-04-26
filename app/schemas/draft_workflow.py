"""Draft workflow response schemas."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.core.enums import ClaimGraphRelationshipType, ClaimReviewAction, StructuredReasoningCategory, SupportStatus
from app.schemas.review_decision import DraftReviewQueueStateRead
from app.schemas.verification import SupportAssessmentRead


class DraftVerdictCountsRead(BaseModel):
    supported: int
    partially_supported: int
    overstated: int
    ambiguous: int
    unsupported: int
    unverified: int


class DraftExcludedLinkRead(BaseModel):
    code: str | None
    message: str | None


class DraftReviewScopeRead(BaseModel):
    scope_kind: str
    allowed_source_document_count: int


class DraftClaimRelationshipRead(BaseModel):
    relationship_type: ClaimGraphRelationshipType
    related_claim_id: UUID
    related_claim_text: str
    reason_code: str | None
    reason_text: str | None
    confidence_score: float | None


class DraftClaimChangeSummaryRead(BaseModel):
    current_verdict: SupportStatus | None
    previous_verdict: SupportStatus | None
    verdict_changed: bool
    current_confidence_score: float | None
    previous_confidence_score: float | None
    confidence_changed: bool
    current_primary_anchor: str | None
    previous_primary_anchor: str | None
    primary_anchor_changed: bool
    support_changed: bool
    current_support_assessment_count: int
    previous_support_assessment_count: int
    current_excluded_link_count: int
    previous_excluded_link_count: int
    current_flags: list[str]
    previous_flags: list[str]
    flags_changed: bool
    current_reasoning_categories: list[StructuredReasoningCategory]
    previous_reasoning_categories: list[StructuredReasoningCategory]
    reasoning_categories_changed: bool
    changed_since_last_run: bool


class DraftFlaggedClaimRead(BaseModel):
    claim_id: UUID
    draft_sequence: int
    claim_text: str
    verdict: SupportStatus
    assertion_context: str | None
    reasoning: str | None
    deterministic_flags: list[str]
    primary_anchor: str | None
    support_assessments: list[SupportAssessmentRead]
    excluded_links: list[DraftExcludedLinkRead]
    scope: DraftReviewScopeRead | None
    suggested_fix: str | None
    confidence_score: float | None
    latest_verification_run_id: UUID | None
    latest_verification_run_at: datetime | None
    reasoning_categories: list[StructuredReasoningCategory]
    changed_since_last_run: bool
    change_summary: DraftClaimChangeSummaryRead | None
    contradiction_flags: list[str]
    claim_relationships: list[DraftClaimRelationshipRead]


class DraftReviewDecisionSummaryRead(BaseModel):
    action: ClaimReviewAction
    note: str | None
    proposed_replacement_text: str | None
    created_at: datetime


class DraftResolvedFlaggedClaimRead(BaseModel):
    claim: DraftFlaggedClaimRead
    latest_decision: DraftReviewDecisionSummaryRead


class DraftCompileResultRead(BaseModel):
    draft_id: UUID
    total_claims: int
    verdict_counts: DraftVerdictCountsRead
    flagged_claims: list[DraftFlaggedClaimRead]


class DraftFlaggedClaimCountsRead(BaseModel):
    unsupported: int
    ambiguous: int
    overstated: int
    unverified: int
    total: int


class DraftReviewIssueBucketsRead(BaseModel):
    unsupported: list[DraftFlaggedClaimRead]
    overstated: list[DraftFlaggedClaimRead]
    ambiguous: list[DraftFlaggedClaimRead]
    unverified: list[DraftFlaggedClaimRead]


class DraftReviewFlagBucketRead(BaseModel):
    flag: str
    claim_count: int
    claims: list[DraftFlaggedClaimRead]


class DraftReviewOverviewRead(BaseModel):
    total_claims: int
    total_flagged_claims: int
    highest_severity_bucket: SupportStatus | None
    top_issue_categories: list[str]


class DraftReviewFreshnessRead(BaseModel):
    state_source: str
    has_persisted_review_runs: bool
    last_review_run_at: datetime | None
    latest_review_run_id: UUID | None
    latest_review_run_status: str | None
    latest_decision_at: datetime | None
    has_decisions_after_latest_run: bool
    latest_claim_verification_at: datetime | None
    latest_verification_run_id: UUID | None
    has_verification_activity_after_latest_run: bool
    is_stale: bool


class DraftReviewRunSummaryRead(BaseModel):
    run_id: UUID
    status: str
    created_at: datetime
    total_claims: int
    total_flagged_claims: int
    resolved_flagged_claims: int
    remaining_flagged_claims: int
    highest_severity_bucket: SupportStatus | None


class DraftWeakSupportClusterRead(BaseModel):
    flag: str
    claim_count: int
    claim_ids: list[UUID]


class DraftReviewIntelligenceSummaryRead(BaseModel):
    risk_distribution: DraftVerdictCountsRead
    most_unstable_claim_ids: list[UUID]
    repeatedly_changed_claim_ids: list[UUID]
    weak_support_claim_ids: list[UUID]
    contradiction_claim_ids: list[UUID]
    contradiction_pair_count: int
    duplicate_pair_count: int
    weak_support_clusters: list[DraftWeakSupportClusterRead]


class DraftReviewResultRead(BaseModel):
    draft_id: UUID
    total_claims: int
    verdict_counts: DraftVerdictCountsRead
    flagged_claim_counts: DraftFlaggedClaimCountsRead
    review_overview: DraftReviewOverviewRead
    freshness: DraftReviewFreshnessRead
    queue_state: DraftReviewQueueStateRead
    active_queue_claims: list[DraftFlaggedClaimRead]
    resolved_claims: list[DraftResolvedFlaggedClaimRead]
    latest_review_run: DraftReviewRunSummaryRead | None = None
    previous_review_run: DraftReviewRunSummaryRead | None = None
    intelligence_summary: DraftReviewIntelligenceSummaryRead | None = None
    issue_buckets: DraftReviewIssueBucketsRead
    flag_buckets: list[DraftReviewFlagBucketRead]
    top_risky_claims: list[DraftFlaggedClaimRead]
    summary: str
