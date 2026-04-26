"""Verification and decision schemas."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.core.enums import DecisionAction, LinkType, StructuredReasoningCategory, SupportStatus
from app.schemas.common import ORMModel


class VerificationRequest(BaseModel):
    model_version: str | None = None
    prompt_version: str | None = None


class SupportAssessmentRead(BaseModel):
    segment_id: UUID
    anchor: str
    role: str
    contribution: str


class VerificationSupportClaimScopeRead(BaseModel):
    claim_id: UUID
    draft_id: UUID
    matter_id: UUID
    evidence_bundle_id: UUID | None
    scope_kind: str
    allowed_source_document_ids: list[UUID]


class VerificationSupportLinkRead(BaseModel):
    link_id: UUID
    claim_id: UUID
    segment_id: UUID
    source_document_id: UUID | None
    sequence_order: int | None
    link_type: LinkType
    citation_text: str | None
    user_confirmed: bool
    anchor: str | None
    evidence_role: str | None


class ExcludedSupportLinkRead(BaseModel):
    link_id: UUID
    claim_id: UUID
    segment_id: UUID
    code: str | None
    message: str | None


class SupportItemRead(BaseModel):
    order: int
    segment_id: UUID
    source_document_id: UUID
    anchor: str
    evidence_role: str
    speaker: str | None
    segment_type: str
    raw_text: str
    normalized_text: str


class VerificationSupportProviderOutputRead(BaseModel):
    primary_anchor: str | None
    support_assessments: list[SupportAssessmentRead]


class VerificationSupportSnapshotRead(BaseModel):
    claim_scope: VerificationSupportClaimScopeRead
    valid_support_links: list[VerificationSupportLinkRead]
    excluded_support_links: list[ExcludedSupportLinkRead]
    support_items: list[SupportItemRead]
    citations: list[str]
    provider_output: VerificationSupportProviderOutputRead


class VerificationRunRead(ORMModel):
    id: UUID
    claim_unit_id: UUID
    model_version: str
    prompt_version: str
    deterministic_flags: list[str]
    reasoning_categories: list[StructuredReasoningCategory]
    verdict: SupportStatus
    reasoning: str
    suggested_fix: str | None
    confidence_score: float | None
    created_at: datetime
    support_snapshot_status: str | None = None
    support_snapshot_note: str | None = None
    support_snapshot_version: int | None = None
    support_snapshot: VerificationSupportSnapshotRead | None = None


class VerificationResultRead(BaseModel):
    id: UUID
    claim_unit_id: UUID
    model_version: str
    prompt_version: str
    deterministic_flags: list[str]
    reasoning_categories: list[StructuredReasoningCategory]
    verdict: SupportStatus
    reasoning: str
    suggested_fix: str | None
    confidence_score: float | None
    created_at: datetime
    primary_anchor: str | None
    support_assessments: list[SupportAssessmentRead]


class UserDecisionCreate(BaseModel):
    action: DecisionAction
    note: str | None = None


class UserDecisionRead(ORMModel):
    id: UUID
    verification_run_id: UUID
    action: DecisionAction
    note: str | None
    created_at: datetime
