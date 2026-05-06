"""Narrow MVP-facing schemas for citation verification."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.core.enums import StructuredReasoningCategory, SupportStatus
from app.schemas.draft import DraftTextCreate


class CitationVerificationRequest(DraftTextCreate):
    """Simple draft-text entrypoint for citation verification MVP."""


class CitationVerdictCountsRead(BaseModel):
    supported: int
    partially_supported: int
    overstated: int
    ambiguous: int
    unsupported: int
    unverified: int
    contradicted: int


class CitationAuthorityStatusCountsRead(BaseModel):
    authority_unverified: int
    citation_recognized: int
    authority_candidate_parsed: int
    authority_matched: int
    linked_authority_support_present: int
    not_reviewed: int


class CitationAuthorityContentStatusCountsRead(BaseModel):
    not_applicable: int
    unavailable: int
    available: int
    support_verified: int


class CitationVerificationSummaryRead(BaseModel):
    total_claims: int
    total_cited_propositions: int
    flagged_citation_count: int
    verdict_counts: CitationVerdictCountsRead
    authority_status_counts: CitationAuthorityStatusCountsRead
    authority_content_status_counts: CitationAuthorityContentStatusCountsRead


class CitationParsedAuthorityRead(BaseModel):
    case_name: str | None
    reporter_volume: str | None
    reporter_abbreviation: str | None
    first_page: str | None
    pin_cite: str | None
    court: str | None
    year: int | None


class CitationSourceSpanRead(BaseModel):
    start: int
    end: int


class CitationMatchedAuthorityRead(BaseModel):
    authority_id: str
    canonical_name: str
    canonical_citation: str
    reporter_volume: str
    reporter_abbreviation: str
    first_page: str
    court: str | None
    year: int | None
    source_name: str


class CitationExternalAuthorityRead(BaseModel):
    provider: str
    provider_cluster_id: str | None
    case_name: str | None
    canonical_citation: str | None
    absolute_url: str | None
    date_filed: str | None
    year: int | None
    normalized_citations: list[str]


class CitationVerificationItemRead(BaseModel):
    claim_id: UUID
    draft_sequence: int
    citation_text: str
    citation_span: CitationSourceSpanRead
    citation_kind: str
    citation_parse_status: str
    proposition_text: str
    assertion_context: str | None
    authority_status: str
    authority_match_status: str
    authority_lookup_status: str
    authority_lookup_provider: str | None
    authority_lookup_error: str | None
    authority_lookup_cached: bool
    parsed_authority: CitationParsedAuthorityRead | None
    normalized_authority_reference: str | None
    matched_authority: CitationMatchedAuthorityRead | None
    external_authority: CitationExternalAuthorityRead | None
    authority_content_status: str
    authority_excerpt: str | None
    support_verification_basis: str | None
    proposition_verdict: SupportStatus
    reasoning: str | None
    reasoning_categories: list[StructuredReasoningCategory]
    confidence_score: float | None
    primary_anchor: str | None
    support_snippet: str | None
    suggested_fix: str | None
    verification_run_id: UUID | None
    verified_at: datetime | None


class CitationVerificationResultRead(BaseModel):
    draft_id: UUID
    matter_id: UUID
    title: str
    review_run_id: UUID | None
    reviewed_at: datetime | None
    summary: CitationVerificationSummaryRead
    citations: list[CitationVerificationItemRead]


class CitationVerificationPdfResultRead(BaseModel):
    pdf_text_status: str
    extracted_character_count: int
    page_count: int | None
    extraction_warnings: list[str]
    extracted_text_preview: str | None
    citation_verification: CitationVerificationResultRead | None
