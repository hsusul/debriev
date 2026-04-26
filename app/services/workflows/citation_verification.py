"""MVP-facing citation verification workflow."""

from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.enums import SupportStatus
from app.models import ClaimUnit, VerificationRun
from app.repositories.claims import ClaimsRepository
from app.schemas.citation_verification import CitationVerificationRequest
from app.services.parsing.case_citations import (
    CASE_CITATION_RE,
    AuthorityMatch,
    ParsedCaseCitation,
    normalize_case_citation_text,
    parse_case_citation,
    resolve_case_authority,
)
from app.services.verification.snapshot_adapter import parse_verification_support_snapshot
from app.services.workflows.draft_intake import DraftIntakeService
from app.services.workflows.draft_review import DraftReviewExecutionService


FLAGGED_VERDICTS = frozenset(
    {
        SupportStatus.OVERSTATED,
        SupportStatus.AMBIGUOUS,
        SupportStatus.UNSUPPORTED,
        SupportStatus.UNVERIFIED,
        SupportStatus.CONTRADICTED,
    }
)


@dataclass(slots=True)
class CitationVerdictCounts:
    supported: int = 0
    partially_supported: int = 0
    overstated: int = 0
    ambiguous: int = 0
    unsupported: int = 0
    unverified: int = 0
    contradicted: int = 0

    def increment(self, verdict: SupportStatus) -> None:
        if verdict == SupportStatus.SUPPORTED:
            self.supported += 1
        elif verdict == SupportStatus.PARTIALLY_SUPPORTED:
            self.partially_supported += 1
        elif verdict == SupportStatus.OVERSTATED:
            self.overstated += 1
        elif verdict == SupportStatus.AMBIGUOUS:
            self.ambiguous += 1
        elif verdict == SupportStatus.UNSUPPORTED:
            self.unsupported += 1
        elif verdict == SupportStatus.UNVERIFIED:
            self.unverified += 1
        elif verdict == SupportStatus.CONTRADICTED:
            self.contradicted += 1


@dataclass(slots=True)
class CitationAuthorityStatusCounts:
    authority_unverified: int = 0
    citation_recognized: int = 0
    authority_candidate_parsed: int = 0
    authority_matched: int = 0
    linked_authority_support_present: int = 0
    not_reviewed: int = 0

    def increment(self, status: str) -> None:
        if status == "citation_recognized":
            self.citation_recognized += 1
            self.authority_unverified += 1
        elif status == "authority_candidate_parsed":
            self.authority_candidate_parsed += 1
            self.authority_unverified += 1
        elif status == "authority_matched":
            self.authority_matched += 1
        elif status == "linked_authority_support_present":
            self.linked_authority_support_present += 1
        elif status == "not_reviewed":
            self.not_reviewed += 1
        else:
            self.authority_unverified += 1


@dataclass(slots=True)
class CitationParsedAuthority:
    case_name: str | None
    reporter_volume: str | None
    reporter_abbreviation: str | None
    first_page: str | None
    court: str | None
    year: int | None


@dataclass(slots=True)
class CitationMatchedAuthority:
    authority_id: str
    canonical_name: str
    canonical_citation: str
    reporter_volume: str
    reporter_abbreviation: str
    first_page: str
    court: str | None
    year: int | None
    source_name: str


@dataclass(slots=True)
class CitationVerificationItem:
    claim_id: UUID
    draft_sequence: int
    citation_text: str
    proposition_text: str
    assertion_context: str | None
    authority_status: str
    authority_match_status: str
    parsed_authority: CitationParsedAuthority | None
    normalized_authority_reference: str | None
    matched_authority: CitationMatchedAuthority | None
    proposition_verdict: SupportStatus
    reasoning: str | None
    reasoning_categories: list[str]
    confidence_score: float | None
    primary_anchor: str | None
    support_snippet: str | None
    suggested_fix: str | None
    verification_run_id: UUID | None
    verified_at: object | None


@dataclass(slots=True)
class CitationVerificationSummary:
    total_claims: int
    total_cited_propositions: int
    flagged_citation_count: int
    verdict_counts: CitationVerdictCounts
    authority_status_counts: CitationAuthorityStatusCounts


@dataclass(slots=True)
class CitationVerificationResult:
    draft_id: UUID
    matter_id: UUID
    title: str
    review_run_id: UUID | None
    reviewed_at: object | None
    summary: CitationVerificationSummary
    citations: list[CitationVerificationItem] = field(default_factory=list)


class CitationVerificationService:
    """Run the narrow MVP citation verification flow on fresh draft text."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.draft_intake = DraftIntakeService(session)
        self.claims = ClaimsRepository(session)
        self.review_execution = DraftReviewExecutionService(session)

    def verify_draft_text(self, payload: CitationVerificationRequest) -> CitationVerificationResult:
        intake_result = self.draft_intake.create_draft_from_text(payload)
        # The draft review execution workflow loads claims through its own read-side
        # session boundary, so the intake artifacts need to be durable first.
        self.session.commit()
        review_result = self.review_execution.execute_review(intake_result.draft_id)
        claims = self.claims.list_by_draft(intake_result.draft_id)

        items: list[CitationVerificationItem] = []
        verdict_counts = CitationVerdictCounts()
        authority_status_counts = CitationAuthorityStatusCounts()

        for draft_sequence, claim in enumerate(claims, start=1):
            proposition_pairs = _extract_citation_proposition_pairs(claim.text)
            if not proposition_pairs:
                continue

            latest_run = _latest_verification_run(claim)
            parsed_snapshot = _parse_support_snapshot(latest_run)
            has_linked_support = _has_linked_support(parsed_snapshot)
            proposition_verdict = latest_run.verdict if latest_run is not None else claim.support_status
            primary_anchor = _derive_primary_anchor(parsed_snapshot)
            support_snippet = _build_support_snippet(parsed_snapshot)
            for citation_text, proposition_text in proposition_pairs:
                parsed_citation = parse_case_citation(citation_text)
                matched_authority = resolve_case_authority(parsed_citation)
                authority_status = _derive_authority_status(
                    parsed_citation,
                    matched_authority,
                    has_linked_support=has_linked_support,
                )
                authority_match_status = _derive_authority_match_status(parsed_citation, matched_authority)
                verdict_counts.increment(proposition_verdict)
                authority_status_counts.increment(authority_status)
                items.append(
                    CitationVerificationItem(
                        claim_id=claim.id,
                        draft_sequence=draft_sequence,
                        citation_text=citation_text,
                        proposition_text=proposition_text,
                        assertion_context=claim.assertion.raw_text if claim.assertion is not None else None,
                        authority_status=authority_status,
                        authority_match_status=authority_match_status,
                        parsed_authority=_build_parsed_authority(parsed_citation),
                        normalized_authority_reference=parsed_citation.normalized_authority_reference,
                        matched_authority=_build_matched_authority(matched_authority),
                        proposition_verdict=proposition_verdict,
                        reasoning=latest_run.reasoning if latest_run is not None else None,
                        reasoning_categories=list(latest_run.reasoning_categories) if latest_run is not None else [],
                        confidence_score=latest_run.confidence_score if latest_run is not None else None,
                        primary_anchor=primary_anchor,
                        support_snippet=support_snippet,
                        suggested_fix=latest_run.suggested_fix if latest_run is not None else None,
                        verification_run_id=latest_run.id if latest_run is not None else None,
                        verified_at=latest_run.created_at if latest_run is not None else None,
                    )
                )

        summary = CitationVerificationSummary(
            total_claims=review_result.total_claims,
            total_cited_propositions=len(items),
            flagged_citation_count=sum(1 for item in items if item.proposition_verdict in FLAGGED_VERDICTS),
            verdict_counts=verdict_counts,
            authority_status_counts=authority_status_counts,
        )
        return CitationVerificationResult(
            draft_id=intake_result.draft_id,
            matter_id=intake_result.matter_id,
            title=intake_result.title,
            review_run_id=review_result.latest_review_run.run_id if review_result.latest_review_run is not None else None,
            reviewed_at=review_result.latest_review_run.created_at if review_result.latest_review_run is not None else None,
            summary=summary,
            citations=items,
        )


def _extract_citation_proposition_pairs(text: str) -> list[tuple[str, str]]:
    matches = list(CASE_CITATION_RE.finditer(text))
    if not matches:
        return []

    pairs: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        next_match_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        start = _find_sentence_start(text, match.start())
        end = _find_sentence_end(text, match.end(), next_match_start)
        proposition_text = text[start:end].strip()
        if not proposition_text:
            proposition_text = text.strip()
        pairs.append((normalize_case_citation_text(match.group(0)), proposition_text))
    return pairs


def _find_sentence_start(text: str, position: int) -> int:
    boundary = 0
    for index in range(position - 1, -1, -1):
        if _is_sentence_terminator(text, index):
            boundary = index + 1
            while boundary < len(text) and text[boundary].isspace():
                boundary += 1
            break
    return boundary


def _find_sentence_end(text: str, citation_end: int, fallback_end: int) -> int:
    for index in range(citation_end, min(fallback_end, len(text))):
        if _is_sentence_terminator(text, index):
            return index + 1
    return fallback_end


def _is_sentence_terminator(text: str, index: int) -> bool:
    if text[index] not in ".?!":
        return False
    if index > 0 and text[index - 1].lower() == "v":
        return False

    cursor = index + 1
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text):
        return True
    return text[cursor].isupper()


def _latest_verification_run(claim: ClaimUnit) -> VerificationRun | None:
    runs = list(getattr(claim, "verification_runs", []))
    if not runs:
        return None
    return max(runs, key=lambda run: ((run.created_at or 0), str(run.id)))


def _parse_support_snapshot(run: VerificationRun | None):
    if run is None:
        return None
    return parse_verification_support_snapshot(run.support_snapshot, run.support_snapshot_version)


def _has_linked_support(parsed_snapshot) -> bool:
    if parsed_snapshot is None:
        return False
    return bool(parsed_snapshot.support_items or parsed_snapshot.valid_support_links)


def _derive_authority_status(
    parsed_citation: ParsedCaseCitation,
    matched_authority: AuthorityMatch | None,
    *,
    has_linked_support: bool,
) -> str:
    if has_linked_support:
        return "linked_authority_support_present"
    if matched_authority is not None:
        return "authority_matched"
    if parsed_citation.has_structured_authority_candidate:
        return "authority_candidate_parsed"
    if parsed_citation.is_case_citation:
        return "citation_recognized"
    return "not_reviewed"


def _derive_authority_match_status(
    parsed_citation: ParsedCaseCitation,
    matched_authority: AuthorityMatch | None,
) -> str:
    if matched_authority is not None:
        return "matched"
    if parsed_citation.has_structured_authority_candidate:
        return "no_match"
    if parsed_citation.is_case_citation:
        return "recognized_only"
    return "not_reviewed"


def _build_parsed_authority(parsed_citation: ParsedCaseCitation) -> CitationParsedAuthority | None:
    if not parsed_citation.is_case_citation:
        return None
    return CitationParsedAuthority(
        case_name=parsed_citation.case_name,
        reporter_volume=parsed_citation.reporter_volume,
        reporter_abbreviation=parsed_citation.reporter_abbreviation,
        first_page=parsed_citation.first_page,
        court=parsed_citation.court,
        year=parsed_citation.year,
    )


def _build_matched_authority(matched_authority: AuthorityMatch | None) -> CitationMatchedAuthority | None:
    if matched_authority is None:
        return None
    return CitationMatchedAuthority(
        authority_id=matched_authority.authority_id,
        canonical_name=matched_authority.canonical_name,
        canonical_citation=matched_authority.canonical_citation,
        reporter_volume=matched_authority.reporter_volume,
        reporter_abbreviation=matched_authority.reporter_abbreviation,
        first_page=matched_authority.first_page,
        court=matched_authority.court,
        year=matched_authority.year,
        source_name=matched_authority.source_name,
    )


def _derive_primary_anchor(parsed_snapshot) -> str | None:
    if parsed_snapshot is None:
        return None
    if parsed_snapshot.provider_output.primary_anchor:
        return parsed_snapshot.provider_output.primary_anchor
    if parsed_snapshot.support_items:
        return parsed_snapshot.support_items[0].anchor
    return None


def _build_support_snippet(parsed_snapshot) -> str | None:
    if parsed_snapshot is None:
        return None
    if parsed_snapshot.provider_output.support_assessments:
        first = parsed_snapshot.provider_output.support_assessments[0]
        return f"{first.anchor}: {first.contribution}"
    if parsed_snapshot.support_items:
        first = parsed_snapshot.support_items[0]
        snippet = first.raw_text.strip()
        if len(snippet) > 220:
            snippet = f"{snippet[:217].rstrip()}..."
        return snippet
    return None
