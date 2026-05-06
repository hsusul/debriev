"""Narrow MVP-facing schemas for case PDF verification."""

from pydantic import BaseModel

from app.core.enums import SupportStatus


class CasePdfAuthorityMetadataRead(BaseModel):
    case_name: str | None
    reporter_volume: str | None
    reporter_abbreviation: str | None
    first_page: str | None
    court: str | None
    year: int | None
    canonical_citation: str | None


class CasePdfVerificationResultRead(BaseModel):
    pdf_text_status: str
    extracted_authority_metadata: CasePdfAuthorityMetadataRead | None
    extracted_character_count: int
    page_count: int | None
    extraction_warnings: list[str]
    extracted_text_preview: str | None
    citation_match_status: str
    statement_verdict: SupportStatus
    reasoning: str | None
    support_snippet: str | None
    confidence_score: float | None
    suggested_fix: str | None
