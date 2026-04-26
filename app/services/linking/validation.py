"""Validation helpers for support-link invariants and legacy-link revalidation."""

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.exceptions import NotFoundError, ValidationError
from app.models import Assertion, ClaimUnit, Draft, Segment, SourceDocument, SupportLink
from app.repositories.evidence_bundles import EvidenceBundleRepository


@dataclass(slots=True)
class ClaimScopeContext:
    """Minimal draft scope context needed to validate a support link."""

    claim_id: UUID
    draft_id: UUID
    matter_id: UUID
    evidence_bundle_id: UUID | None


@dataclass(slots=True)
class SegmentScopeContext:
    """Minimal segment scope context needed to validate a support link."""

    segment_id: UUID
    source_document_id: UUID
    matter_id: UUID


@dataclass(slots=True)
class LinkValidationContext:
    """Resolved claim/segment context used to validate a support link."""

    claim: ClaimScopeContext
    segment: SegmentScopeContext


@dataclass(slots=True)
class LinkRevalidationResult:
    """Structured result for checking whether an existing support link is still valid."""

    link_id: UUID
    claim_id: UUID
    segment_id: UUID
    is_valid: bool
    code: str | None = None
    message: str | None = None


@dataclass(slots=True)
class LinkInvalidation:
    """Internal invalidation descriptor reused by write-time and revalidation paths."""

    code: str
    message: str


class LinkValidationService:
    """Centralized validation for support-link trust invariants."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.evidence_bundles = EvidenceBundleRepository(session)

    def validate_link_invariants(self, claim_id: UUID, segment_id: UUID) -> LinkValidationContext:
        claim = self.get_claim_scope_context(claim_id)
        if claim is None:
            raise NotFoundError("Claim unit not found.")

        segment = self._load_segment_scope_context(segment_id)
        if segment is None:
            raise NotFoundError("Segment not found.")

        invalidation = self._detect_invalidation(claim, segment)
        if invalidation is not None:
            raise ValidationError(invalidation.message)

        return LinkValidationContext(claim=claim, segment=segment)

    def get_claim_scope_context(self, claim_id: UUID) -> ClaimScopeContext | None:
        return self._load_claim_scope_context(claim_id)

    def revalidate_link(self, link_id: UUID, claim_id: UUID, segment_id: UUID) -> LinkRevalidationResult:
        claim = self._load_claim_scope_context(claim_id)
        if claim is None:
            return LinkRevalidationResult(
                link_id=link_id,
                claim_id=claim_id,
                segment_id=segment_id,
                is_valid=False,
                code="missing_claim_support_link",
                message="Support link points to a missing claim unit.",
            )

        segment = self._load_segment_scope_context(segment_id)
        if segment is None:
            return LinkRevalidationResult(
                link_id=link_id,
                claim_id=claim_id,
                segment_id=segment_id,
                is_valid=False,
                code="missing_segment_support_link",
                message="Support link points to a missing segment.",
            )

        invalidation = self._detect_invalidation(claim, segment)
        if invalidation is not None:
            return LinkRevalidationResult(
                link_id=link_id,
                claim_id=claim_id,
                segment_id=segment_id,
                is_valid=False,
                code=invalidation.code,
                message=invalidation.message,
            )

        return LinkRevalidationResult(
            link_id=link_id,
            claim_id=claim_id,
            segment_id=segment_id,
            is_valid=True,
        )

    def revalidate_links(self, links: list[SupportLink]) -> list[LinkRevalidationResult]:
        results: list[LinkRevalidationResult] = []
        claim_cache: dict[UUID, ClaimScopeContext | None] = {}
        segment_cache: dict[UUID, SegmentScopeContext | None] = {}

        for link in links:
            link_id = getattr(link, "id")
            claim_id = getattr(link, "claim_unit_id")
            segment_id = getattr(link, "segment_id")

            claim = claim_cache.get(claim_id)
            if claim_id not in claim_cache:
                claim = self._load_claim_scope_context(claim_id)
                claim_cache[claim_id] = claim

            if claim is None:
                results.append(
                    LinkRevalidationResult(
                        link_id=link_id,
                        claim_id=claim_id,
                        segment_id=segment_id,
                        is_valid=False,
                        code="missing_claim_support_link",
                        message="Support link points to a missing claim unit.",
                    )
                )
                continue

            segment = segment_cache.get(segment_id)
            if segment_id not in segment_cache:
                segment = self._load_segment_scope_context(segment_id)
                segment_cache[segment_id] = segment

            if segment is None:
                results.append(
                    LinkRevalidationResult(
                        link_id=link_id,
                        claim_id=claim_id,
                        segment_id=segment_id,
                        is_valid=False,
                        code="missing_segment_support_link",
                        message="Support link points to a missing segment.",
                    )
                )
                continue

            invalidation = self._detect_invalidation(claim, segment)
            if invalidation is not None:
                results.append(
                    LinkRevalidationResult(
                        link_id=link_id,
                        claim_id=claim_id,
                        segment_id=segment_id,
                        is_valid=False,
                        code=invalidation.code,
                        message=invalidation.message,
                    )
                )
                continue

            results.append(
                LinkRevalidationResult(
                    link_id=link_id,
                    claim_id=claim_id,
                    segment_id=segment_id,
                    is_valid=True,
                )
            )

        return results

    def _load_claim_scope_context(self, claim_id: UUID) -> ClaimScopeContext | None:
        stmt = (
            select(
                ClaimUnit.id,
                Draft.id.label("draft_id"),
                Draft.matter_id,
                Draft.evidence_bundle_id,
            )
            .join(Assertion, ClaimUnit.assertion_id == Assertion.id)
            .join(Draft, Assertion.draft_id == Draft.id)
            .where(ClaimUnit.id == claim_id)
        )
        row = self.session.execute(stmt).one_or_none()
        if row is None:
            return None

        return ClaimScopeContext(
            claim_id=row.id,
            draft_id=row.draft_id,
            matter_id=row.matter_id,
            evidence_bundle_id=row.evidence_bundle_id,
        )

    def _load_segment_scope_context(self, segment_id: UUID) -> SegmentScopeContext | None:
        stmt = (
            select(
                Segment.id,
                Segment.source_document_id,
                SourceDocument.matter_id,
            )
            .join(SourceDocument, Segment.source_document_id == SourceDocument.id)
            .where(Segment.id == segment_id)
        )
        row = self.session.execute(stmt).one_or_none()
        if row is None:
            return None

        return SegmentScopeContext(
            segment_id=row.id,
            source_document_id=row.source_document_id,
            matter_id=row.matter_id,
        )

    def _detect_invalidation(
        self,
        claim: ClaimScopeContext,
        segment: SegmentScopeContext,
    ) -> LinkInvalidation | None:
        if claim.matter_id != segment.matter_id:
            return LinkInvalidation(
                code="cross_matter_support_link",
                message="Claim and segment must belong to the same matter.",
            )

        if claim.evidence_bundle_id is not None and not self.evidence_bundles.contains_source_document(
            claim.evidence_bundle_id,
            segment.source_document_id,
        ):
            return LinkInvalidation(
                code="out_of_scope_support_link",
                message="Segment source document is outside the draft evidence bundle.",
            )

        return None
