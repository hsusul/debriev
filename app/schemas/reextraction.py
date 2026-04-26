"""Assertion re-extraction compare/apply schemas."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel

from app.core.enums import ClaimType


class ReExtractionCompareRequest(BaseModel):
    mode: Literal["auto", "legacy", "structured"] = "structured"


class ReExtractionApplyRequest(BaseModel):
    mode: Literal["auto", "legacy", "structured"] = "structured"


class ReExtractionClaimPreviewRead(BaseModel):
    claim_id: UUID | None
    text: str
    normalized_text: str
    claim_type: ClaimType
    sequence_order: int


class ExistingExtractionMetadataRead(BaseModel):
    status: str
    strategy: str | None
    version: int | None
    snapshot_present: bool


class ProposedExtractionMetadataRead(BaseModel):
    strategy: str
    version: int


class ExtractionSnapshotSummaryRead(BaseModel):
    present: bool
    claim_count: int


class AssertionReExtractionComparisonRead(BaseModel):
    assertion_id: UUID
    existing_metadata: ExistingExtractionMetadataRead
    proposed_metadata: ProposedExtractionMetadataRead
    existing_claims: list[ReExtractionClaimPreviewRead]
    proposed_claims: list[ReExtractionClaimPreviewRead]
    materially_changed: bool
    apply_requires_replacement: bool
    can_apply: bool
    blocked_reasons: list[str]


class AssertionReExtractionApplyRead(BaseModel):
    assertion_id: UUID
    applied_strategy: str
    applied_version: int
    materially_changed: bool
    apply_requires_replacement: bool
    claims_replaced: bool
    metadata_updated: bool
    resulting_claims: list[ReExtractionClaimPreviewRead]
    updated_metadata: ExistingExtractionMetadataRead
    snapshot_summary: ExtractionSnapshotSummaryRead
    notes: list[str]


class DraftReExtractionPreviewItemRead(BaseModel):
    assertion_id: UUID
    paragraph_index: int | None
    sentence_index: int | None
    assertion_text: str
    status: Literal["ready", "unchanged", "blocked"]
    existing_metadata: ExistingExtractionMetadataRead
    proposed_metadata: ProposedExtractionMetadataRead
    materially_changed: bool
    apply_requires_replacement: bool
    can_apply: bool
    blocked_reasons: list[str]
    existing_claim_count: int
    proposed_claim_count: int


class DraftReExtractionPreviewRead(BaseModel):
    run_id: UUID
    draft_id: UUID
    requested_mode: Literal["auto", "legacy", "structured"]
    total_assertions: int
    ready_assertions: int
    unchanged_assertions: int
    blocked_assertions: int
    materially_changed_assertions: int
    legacy_unversioned_assertions: int
    items: list[DraftReExtractionPreviewItemRead]


class DraftReExtractionApplyItemRead(BaseModel):
    assertion_id: UUID
    paragraph_index: int | None
    sentence_index: int | None
    assertion_text: str
    status: Literal["applied", "skipped", "blocked"]
    existing_metadata: ExistingExtractionMetadataRead
    proposed_metadata: ProposedExtractionMetadataRead
    materially_changed: bool
    apply_requires_replacement: bool
    can_apply: bool
    claims_replaced: bool
    metadata_updated: bool
    blocked_reasons: list[str]
    resulting_claims: list[ReExtractionClaimPreviewRead]
    notes: list[str]


class DraftReExtractionApplyRead(BaseModel):
    run_id: UUID
    draft_id: UUID
    requested_mode: Literal["auto", "legacy", "structured"]
    total_assertions: int
    applied_assertions: int
    skipped_assertions: int
    blocked_assertions: int
    replaced_assertions: int
    metadata_only_assertions: int
    items: list[DraftReExtractionApplyItemRead]
