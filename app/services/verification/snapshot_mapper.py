"""Shared read-side mappers for adapted verification support snapshots."""

from dataclasses import dataclass, field

from app.schemas.verification import (
    ExcludedSupportLinkRead,
    SupportAssessmentRead,
    SupportItemRead,
    VerificationSupportClaimScopeRead,
    VerificationSupportLinkRead,
    VerificationSupportProviderOutputRead,
    VerificationSupportSnapshotRead,
)
from app.services.verification.snapshot_adapter import ParsedVerificationSupportSnapshot, SnapshotReadStatus


@dataclass(slots=True)
class VerificationHistorySnapshotFields:
    """Stable history response fields derived from a parsed persisted snapshot."""

    support_snapshot_status: SnapshotReadStatus
    support_snapshot_note: str | None
    support_snapshot_version: int | None
    support_snapshot: VerificationSupportSnapshotRead | None

    def as_response_fields(self) -> dict[str, object]:
        return {
            "support_snapshot_status": self.support_snapshot_status,
            "support_snapshot_note": self.support_snapshot_note,
            "support_snapshot_version": self.support_snapshot_version,
            "support_snapshot": (
                self.support_snapshot.model_dump(mode="json") if self.support_snapshot is not None else None
            ),
        }


@dataclass(slots=True)
class AuditSupportAssessmentView:
    anchor: str
    role: str
    contribution: str


@dataclass(slots=True)
class AuditExcludedSupportLinkView:
    code: str | None
    message: str | None


@dataclass(slots=True)
class AuditSnapshotView:
    """Audit/report projection derived from a parsed persisted snapshot."""

    support_snapshot_version: int | None = None
    scope_kind: str | None = None
    allowed_source_document_count: int = 0
    primary_anchor: str | None = None
    support_assessments: list[AuditSupportAssessmentView] = field(default_factory=list)
    excluded_support_links: list[AuditExcludedSupportLinkView] = field(default_factory=list)
    snapshot_note: str | None = None


def map_parsed_snapshot_to_history_fields(
    parsed_snapshot: ParsedVerificationSupportSnapshot,
) -> VerificationHistorySnapshotFields:
    """Shape a parsed snapshot into stable verification history response fields."""

    return VerificationHistorySnapshotFields(
        support_snapshot_status=parsed_snapshot.status,
        support_snapshot_note=parsed_snapshot.note,
        support_snapshot_version=parsed_snapshot.version,
        support_snapshot=_build_history_support_snapshot(parsed_snapshot),
    )


def map_parsed_snapshot_to_audit_view(
    parsed_snapshot: ParsedVerificationSupportSnapshot,
) -> AuditSnapshotView | None:
    """Shape a parsed snapshot into a compact audit-oriented view."""

    if parsed_snapshot.status == "missing":
        return None

    allowed_source_document_count = (
        len(parsed_snapshot.claim_scope.allowed_source_document_ids)
        if parsed_snapshot.claim_scope is not None
        else 0
    )
    return AuditSnapshotView(
        support_snapshot_version=parsed_snapshot.version,
        scope_kind=parsed_snapshot.claim_scope.scope_kind if parsed_snapshot.claim_scope is not None else None,
        allowed_source_document_count=allowed_source_document_count,
        primary_anchor=parsed_snapshot.provider_output.primary_anchor,
        support_assessments=[
            AuditSupportAssessmentView(
                anchor=assessment.anchor,
                role=assessment.role,
                contribution=assessment.contribution,
            )
            for assessment in parsed_snapshot.provider_output.support_assessments
            if assessment.anchor and assessment.role and assessment.contribution
        ],
        excluded_support_links=[
            AuditExcludedSupportLinkView(
                code=link.code,
                message=link.message,
            )
            for link in parsed_snapshot.excluded_support_links
        ],
        snapshot_note=parsed_snapshot.note,
    )


def _build_history_support_snapshot(
    parsed_snapshot: ParsedVerificationSupportSnapshot,
) -> VerificationSupportSnapshotRead | None:
    if parsed_snapshot.status not in {"versioned_v1", "legacy_unversioned_v1"}:
        return None

    claim_scope = parsed_snapshot.claim_scope
    if claim_scope is None:
        return None

    return VerificationSupportSnapshotRead(
        claim_scope=VerificationSupportClaimScopeRead(
            claim_id=claim_scope.claim_id,
            draft_id=claim_scope.draft_id,
            matter_id=claim_scope.matter_id,
            evidence_bundle_id=claim_scope.evidence_bundle_id,
            scope_kind=claim_scope.scope_kind,
            allowed_source_document_ids=claim_scope.allowed_source_document_ids,
        ),
        valid_support_links=[
            VerificationSupportLinkRead(
                link_id=link.link_id,
                claim_id=link.claim_id,
                segment_id=link.segment_id,
                source_document_id=link.source_document_id,
                sequence_order=link.sequence_order,
                link_type=link.link_type,
                citation_text=link.citation_text,
                user_confirmed=link.user_confirmed,
                anchor=link.anchor,
                evidence_role=link.evidence_role,
            )
            for link in parsed_snapshot.valid_support_links
        ],
        excluded_support_links=[
            ExcludedSupportLinkRead(
                link_id=link.link_id,
                claim_id=link.claim_id,
                segment_id=link.segment_id,
                code=link.code,
                message=link.message,
            )
            for link in parsed_snapshot.excluded_support_links
        ],
        support_items=[
            SupportItemRead(
                order=item.order,
                segment_id=item.segment_id,
                source_document_id=item.source_document_id,
                anchor=item.anchor,
                evidence_role=item.evidence_role,
                speaker=item.speaker,
                segment_type=item.segment_type,
                raw_text=item.raw_text,
                normalized_text=item.normalized_text,
            )
            for item in parsed_snapshot.support_items
        ],
        citations=list(parsed_snapshot.citations),
        provider_output=VerificationSupportProviderOutputRead(
            primary_anchor=parsed_snapshot.provider_output.primary_anchor,
            support_assessments=[
                SupportAssessmentRead(
                    segment_id=assessment.segment_id,
                    anchor=assessment.anchor,
                    role=assessment.role,
                    contribution=assessment.contribution,
                )
                for assessment in parsed_snapshot.provider_output.support_assessments
            ],
        ),
    )
