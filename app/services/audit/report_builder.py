"""Audit report helpers."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.core.enums import SupportStatus
from app.models import UserDecision, VerificationRun
from app.services.verification.snapshot_adapter import parse_verification_support_snapshot
from app.services.verification.snapshot_mapper import map_parsed_snapshot_to_audit_view

if TYPE_CHECKING:
    from app.services.workflows.draft_compile import DraftCompileResult
    from app.services.workflows.draft_review import DraftReviewResult


@dataclass(slots=True)
class AuditSupportAssessment:
    """Minimal support-item detail used by audit summaries."""

    anchor: str
    role: str
    contribution: str


@dataclass(slots=True)
class AuditExcludedSupportLink:
    """Excluded support-link detail used by audit summaries."""

    code: str | None
    message: str | None


@dataclass(slots=True)
class AuditVerificationSnapshot:
    """Structured verification detail normalized for audit rendering."""

    verdict: SupportStatus
    deterministic_flags: list[str]
    support_snapshot_version: int | None = None
    scope_kind: str | None = None
    allowed_source_document_count: int = 0
    primary_anchor: str | None = None
    support_assessments: list[AuditSupportAssessment] = field(default_factory=list)
    excluded_support_links: list[AuditExcludedSupportLink] = field(default_factory=list)
    snapshot_note: str | None = None


def build_audit_summary(
    run: VerificationRun,
    decisions: Sequence[UserDecision],
    *,
    verification_result: object | None = None,
) -> str:
    """Build a compact human-readable audit summary for a verification run."""

    snapshot = _build_verification_snapshot(run, verification_result=verification_result)
    parts = [
        f"Verdict: {snapshot.verdict.value}.",
        f"Flags: {', '.join(snapshot.deterministic_flags) if snapshot.deterministic_flags else 'none'}.",
    ]

    if snapshot.snapshot_note:
        parts.append(snapshot.snapshot_note)

    if snapshot.scope_kind:
        parts.append(
            f"Scope: {snapshot.scope_kind} ({snapshot.allowed_source_document_count} allowed source document(s))."
        )

    if snapshot.primary_anchor:
        parts.append(f"Primary support anchor: {snapshot.primary_anchor}.")

    support_summary = _render_support_assessments(snapshot.support_assessments)
    if support_summary:
        parts.append(f"Support items: {support_summary}.")

    excluded_summary = _render_excluded_support_links(snapshot.excluded_support_links)
    if excluded_summary:
        parts.append(f"Excluded links: {excluded_summary}.")

    parts.append(f"User decisions: {_render_user_decisions(decisions)}.")
    return " ".join(parts)


def build_draft_compile_summary(result: "DraftCompileResult") -> str:
    """Build a compact human-readable summary for a draft compile result."""

    counts = result.counts
    parts = [
        f"Draft {result.draft_id} compile summary.",
        f"Total claims: {result.total_claims}.",
        (
            "Verdicts: "
            f"supported={counts.supported}, "
            f"partially_supported={counts.partially_supported}, "
            f"overstated={counts.overstated}, "
            f"ambiguous={counts.ambiguous}, "
            f"unsupported={counts.unsupported}, "
            f"unverified={counts.unverified}."
        ),
    ]

    if not result.flagged_claims:
        parts.append("Flagged claims: none.")
        return " ".join(parts)

    flagged_summary = "; ".join(
        _render_flagged_claim_summary(claim)
        for claim in result.flagged_claims
    )
    parts.append(f"Flagged claims: {flagged_summary}.")
    return " ".join(parts)


def build_draft_review_summary(result: "DraftReviewResult") -> str:
    """Build a compact human-readable summary for a draft review result."""

    flagged_counts = result.flagged_claim_counts
    overview = result.review_overview
    parts = [
        f"Draft {result.draft_id} review summary.",
        f"Total claims: {result.total_claims}.",
        f"Flagged claims: {flagged_counts.total}.",
        (
            "Review buckets: "
            f"unsupported={flagged_counts.unsupported}, "
            f"overstated={flagged_counts.overstated}, "
            f"ambiguous={flagged_counts.ambiguous}, "
            f"unverified={flagged_counts.unverified}."
        ),
    ]

    if overview.highest_severity_bucket is not None:
        parts.append(f"Highest severity: {overview.highest_severity_bucket.value.lower()}.")

    if overview.top_issue_categories:
        parts.append(f"Top issue categories: {', '.join(overview.top_issue_categories)}.")

    if result.flag_buckets:
        common_flags = ", ".join(
            f"{bucket.flag}={bucket.claim_count}"
            for bucket in result.flag_buckets[:3]
        )
        parts.append(f"Common flags: {common_flags}.")

    if result.top_risky_claims:
        top_targets = "; ".join(
            _render_flagged_claim_summary(claim)
            for claim in result.top_risky_claims[:3]
        )
        parts.append(f"Top review targets: {top_targets}.")

    return " ".join(parts)


def _build_verification_snapshot(
    run: VerificationRun,
    *,
    verification_result: object | None,
) -> AuditVerificationSnapshot:
    parsed_snapshot = parse_verification_support_snapshot(run.support_snapshot, run.support_snapshot_version)
    mapped_snapshot = map_parsed_snapshot_to_audit_view(parsed_snapshot)
    if mapped_snapshot is not None:
        return AuditVerificationSnapshot(
            verdict=run.verdict,
            deterministic_flags=list(run.deterministic_flags),
            support_snapshot_version=mapped_snapshot.support_snapshot_version,
            scope_kind=mapped_snapshot.scope_kind,
            allowed_source_document_count=mapped_snapshot.allowed_source_document_count,
            primary_anchor=mapped_snapshot.primary_anchor,
            support_assessments=[
                AuditSupportAssessment(
                    anchor=assessment.anchor,
                    role=assessment.role,
                    contribution=assessment.contribution,
                )
                for assessment in mapped_snapshot.support_assessments
            ],
            excluded_support_links=[
                AuditExcludedSupportLink(
                    code=entry.code,
                    message=entry.message,
                )
                for entry in mapped_snapshot.excluded_support_links
            ],
            snapshot_note=mapped_snapshot.snapshot_note,
        )

    if verification_result is None:
        return AuditVerificationSnapshot(
            verdict=run.verdict,
            deterministic_flags=list(run.deterministic_flags),
        )

    raw_assessments = getattr(verification_result, "support_assessments", []) or []
    support_assessments = [
        AuditSupportAssessment(
            anchor=assessment.anchor,
            role=assessment.role,
            contribution=assessment.contribution,
        )
        for assessment in raw_assessments
    ]

    return AuditVerificationSnapshot(
        verdict=getattr(verification_result, "verdict", run.verdict),
        deterministic_flags=list(getattr(verification_result, "deterministic_flags", run.deterministic_flags)),
        primary_anchor=getattr(verification_result, "primary_anchor", None),
        support_assessments=support_assessments,
    )


def _render_support_assessments(assessments: Sequence[AuditSupportAssessment]) -> str:
    if not assessments:
        return ""
    return "; ".join(
        f"{assessment.anchor} [{assessment.role}]: {assessment.contribution}" for assessment in assessments
    )


def _render_excluded_support_links(excluded_links: Sequence[AuditExcludedSupportLink]) -> str:
    if not excluded_links:
        return ""

    counts: dict[tuple[str | None, str | None], int] = {}
    for link in excluded_links:
        key = (link.code, link.message)
        counts[key] = counts.get(key, 0) + 1

    parts: list[str] = []
    for (code, message), count in counts.items():
        label = code or "invalid_support_link"
        if message:
            parts.append(f"{label} x{count} ({message})")
        else:
            parts.append(f"{label} x{count}")
    return "; ".join(parts)


def _render_user_decisions(decisions: Sequence[UserDecision]) -> str:
    if not decisions:
        return "none"

    return ", ".join(
        f"{decision.action.value} ({decision.note})" if decision.note else decision.action.value for decision in decisions
    )


def _render_flagged_claim_summary(claim: object) -> str:
    flags = getattr(claim, "deterministic_flags", []) or []
    primary_anchor = getattr(claim, "primary_anchor", None)
    anchor_label = primary_anchor or "no primary anchor"
    flag_label = ", ".join(flags) if flags else "no flags"
    return (
        f"{getattr(claim, 'verdict').value} "
        f"{getattr(claim, 'claim_id')} "
        f"@ {anchor_label} "
        f"[{flag_label}]"
    )
