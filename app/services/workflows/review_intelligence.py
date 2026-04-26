"""Shared review-intelligence helpers for claim graph, diffs, and instability signals."""

from collections import Counter
from dataclasses import dataclass, field
from uuid import UUID

from app.core.enums import ClaimGraphRelationshipType, StructuredReasoningCategory, SupportStatus
from app.models import ClaimGraphEdge, ClaimUnit, VerificationRun
from app.services.verification.snapshot_adapter import parse_verification_support_snapshot


@dataclass(slots=True)
class ClaimRunSummary:
    claim_id: UUID
    verdict: SupportStatus | None
    confidence_score: float | None
    deterministic_flags: list[str] = field(default_factory=list)
    reasoning_categories: list[str] = field(default_factory=list)
    primary_anchor: str | None = None
    support_assessment_count: int = 0
    excluded_link_count: int = 0
    latest_verification_run_id: UUID | None = None


@dataclass(slots=True)
class ClaimChangeSummary:
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
    current_flags: list[str] = field(default_factory=list)
    previous_flags: list[str] = field(default_factory=list)
    flags_changed: bool = False
    current_reasoning_categories: list[str] = field(default_factory=list)
    previous_reasoning_categories: list[str] = field(default_factory=list)
    reasoning_categories_changed: bool = False
    changed_since_last_run: bool = False


@dataclass(slots=True)
class ClaimGraphRelationshipView:
    relationship_type: ClaimGraphRelationshipType
    related_claim_id: UUID
    related_claim_text: str
    reason_code: str | None = None
    reason_text: str | None = None
    confidence_score: float | None = None


@dataclass(slots=True)
class DraftWeakSupportCluster:
    flag: str
    claim_count: int
    claim_ids: list[UUID] = field(default_factory=list)


@dataclass(slots=True)
class DraftReviewIntelligenceSummary:
    risk_distribution: dict[str, int]
    most_unstable_claim_ids: list[UUID] = field(default_factory=list)
    repeatedly_changed_claim_ids: list[UUID] = field(default_factory=list)
    weak_support_claim_ids: list[UUID] = field(default_factory=list)
    contradiction_claim_ids: list[UUID] = field(default_factory=list)
    contradiction_pair_count: int = 0
    duplicate_pair_count: int = 0
    weak_support_clusters: list[DraftWeakSupportCluster] = field(default_factory=list)


def build_claim_run_summary_from_verification_run(
    claim_id: UUID,
    run: VerificationRun | None,
    *,
    fallback_verdict: SupportStatus | None,
) -> ClaimRunSummary:
    if run is None:
        return ClaimRunSummary(
            claim_id=claim_id,
            verdict=fallback_verdict,
            confidence_score=None,
        )

    parsed_snapshot = parse_verification_support_snapshot(run.support_snapshot, run.support_snapshot_version)
    return ClaimRunSummary(
        claim_id=claim_id,
        verdict=run.verdict,
        confidence_score=run.confidence_score,
        deterministic_flags=list(run.deterministic_flags),
        reasoning_categories=list(run.reasoning_categories),
        primary_anchor=parsed_snapshot.provider_output.primary_anchor,
        support_assessment_count=len(parsed_snapshot.provider_output.support_assessments),
        excluded_link_count=len(parsed_snapshot.excluded_support_links),
        latest_verification_run_id=run.id,
    )


def build_claim_run_summary_from_snapshot_entry(entry: dict[str, object] | None) -> ClaimRunSummary | None:
    if not isinstance(entry, dict):
        return None
    claim_id_value = entry.get("claim_id")
    if not isinstance(claim_id_value, str):
        return None
    try:
        claim_id = UUID(claim_id_value)
    except ValueError:
        return None

    verdict_value = entry.get("verdict")
    verdict: SupportStatus | None = None
    if isinstance(verdict_value, str):
        try:
            verdict = SupportStatus(verdict_value)
        except ValueError:
            verdict = None
    verification_run_id_value = entry.get("latest_verification_run_id")
    latest_verification_run_id: UUID | None = None
    if isinstance(verification_run_id_value, str):
        try:
            latest_verification_run_id = UUID(verification_run_id_value)
        except ValueError:
            latest_verification_run_id = None

    return ClaimRunSummary(
        claim_id=claim_id,
        verdict=verdict,
        confidence_score=_float_value(entry.get("confidence_score")),
        deterministic_flags=_string_list(entry.get("deterministic_flags")),
        reasoning_categories=_string_list(entry.get("reasoning_categories")),
        primary_anchor=entry.get("primary_anchor") if isinstance(entry.get("primary_anchor"), str) else None,
        support_assessment_count=_int_value(entry.get("support_assessment_count")),
        excluded_link_count=_int_value(entry.get("excluded_link_count")),
        latest_verification_run_id=latest_verification_run_id,
    )


def build_claim_change_summary(
    current: ClaimRunSummary,
    previous: ClaimRunSummary | None,
) -> ClaimChangeSummary:
    if previous is None:
        return ClaimChangeSummary(
            current_verdict=current.verdict,
            previous_verdict=None,
            verdict_changed=False,
            current_confidence_score=current.confidence_score,
            previous_confidence_score=None,
            confidence_changed=False,
            current_primary_anchor=current.primary_anchor,
            previous_primary_anchor=None,
            primary_anchor_changed=False,
            support_changed=False,
            current_support_assessment_count=current.support_assessment_count,
            previous_support_assessment_count=0,
            current_excluded_link_count=current.excluded_link_count,
            previous_excluded_link_count=0,
            current_flags=list(current.deterministic_flags),
            previous_flags=[],
            flags_changed=False,
            current_reasoning_categories=list(current.reasoning_categories),
            previous_reasoning_categories=[],
            reasoning_categories_changed=False,
            changed_since_last_run=False,
        )

    flags_changed = set(current.deterministic_flags) != set(previous.deterministic_flags)
    reasoning_categories_changed = set(current.reasoning_categories) != set(previous.reasoning_categories)
    primary_anchor_changed = current.primary_anchor != previous.primary_anchor
    support_changed = (
        primary_anchor_changed
        or current.support_assessment_count != previous.support_assessment_count
        or current.excluded_link_count != previous.excluded_link_count
    )
    confidence_changed = current.confidence_score != previous.confidence_score
    verdict_changed = current.verdict != previous.verdict
    changed_since_last_run = any(
        (
            verdict_changed,
            confidence_changed,
            flags_changed,
            support_changed,
            reasoning_categories_changed,
        )
    )

    return ClaimChangeSummary(
        current_verdict=current.verdict,
        previous_verdict=previous.verdict,
        verdict_changed=verdict_changed,
        current_confidence_score=current.confidence_score,
        previous_confidence_score=previous.confidence_score,
        confidence_changed=confidence_changed,
        current_primary_anchor=current.primary_anchor,
        previous_primary_anchor=previous.primary_anchor,
        primary_anchor_changed=primary_anchor_changed,
        support_changed=support_changed,
        current_support_assessment_count=current.support_assessment_count,
        previous_support_assessment_count=previous.support_assessment_count,
        current_excluded_link_count=current.excluded_link_count,
        previous_excluded_link_count=previous.excluded_link_count,
        current_flags=list(current.deterministic_flags),
        previous_flags=list(previous.deterministic_flags),
        flags_changed=flags_changed,
        current_reasoning_categories=list(current.reasoning_categories),
        previous_reasoning_categories=list(previous.reasoning_categories),
        reasoning_categories_changed=reasoning_categories_changed,
        changed_since_last_run=changed_since_last_run,
    )


def build_claim_graph_relationship_index(
    edges: list[ClaimGraphEdge],
    *,
    claim_text_by_id: dict[UUID, str],
) -> dict[UUID, list[ClaimGraphRelationshipView]]:
    relationship_index: dict[UUID, list[ClaimGraphRelationshipView]] = {}
    for edge in edges:
        relationship_index.setdefault(edge.source_claim_id, []).append(
            ClaimGraphRelationshipView(
                relationship_type=edge.relationship_type,
                related_claim_id=edge.target_claim_id,
                related_claim_text=claim_text_by_id.get(edge.target_claim_id, "Unknown claim"),
                reason_code=edge.reason_code,
                reason_text=edge.reason_text,
                confidence_score=edge.confidence_score,
            )
        )

    for relationships in relationship_index.values():
        relationships.sort(
            key=lambda item: (
                item.relationship_type.value,
                item.related_claim_text,
                str(item.related_claim_id),
            )
        )
    return relationship_index


def build_claim_contradiction_flags(
    *,
    relationships: list[ClaimGraphRelationshipView],
    verification_runs: list[VerificationRun],
) -> list[str]:
    flags: list[str] = []
    if any(relationship.relationship_type == ClaimGraphRelationshipType.CONTRADICTS for relationship in relationships):
        flags.append("cross_claim_contradiction")

    verdicts = {run.verdict for run in verification_runs}
    if len(verdicts) > 1:
        flags.append("run_verdict_instability")

    return flags


def build_draft_review_intelligence_summary(
    *,
    claims: list[ClaimUnit],
    current_summaries_by_claim: dict[UUID, ClaimRunSummary],
    change_summaries_by_claim: dict[UUID, ClaimChangeSummary],
    contradiction_flags_by_claim: dict[UUID, list[str]],
    relationship_index: dict[UUID, list[ClaimGraphRelationshipView]],
    edges: list[ClaimGraphEdge],
    risk_distribution: dict[str, int],
) -> DraftReviewIntelligenceSummary:
    repeatedly_changed_claim_ids = [
        claim.id
        for claim in claims
        if len({run.verdict for run in claim.verification_runs}) > 1
    ]

    scored_unstable_claims = sorted(
        (
            (
                _change_score(change_summaries_by_claim.get(claim.id)),
                str(claim.id),
                claim.id,
            )
            for claim in claims
        ),
        reverse=True,
    )
    most_unstable_claim_ids = [
        claim_id
        for score, _, claim_id in scored_unstable_claims
        if score > 0
    ][:5]

    weak_support_claim_ids = [
        claim_id
        for claim_id, summary in current_summaries_by_claim.items()
        if StructuredReasoningCategory.WEAK_SUPPORT.value in summary.reasoning_categories
    ]
    contradiction_claim_ids = [
        claim_id
        for claim_id, flags in contradiction_flags_by_claim.items()
        if flags
    ]

    weak_support_clusters = _build_weak_support_clusters(
        current_summaries_by_claim=current_summaries_by_claim,
        weak_support_claim_ids=weak_support_claim_ids,
    )
    contradiction_pair_count = _pair_count(edges, ClaimGraphRelationshipType.CONTRADICTS)
    duplicate_pair_count = _pair_count(edges, ClaimGraphRelationshipType.DUPLICATE_OF)

    return DraftReviewIntelligenceSummary(
        risk_distribution=risk_distribution,
        most_unstable_claim_ids=most_unstable_claim_ids,
        repeatedly_changed_claim_ids=repeatedly_changed_claim_ids,
        weak_support_claim_ids=weak_support_claim_ids,
        contradiction_claim_ids=contradiction_claim_ids,
        contradiction_pair_count=contradiction_pair_count,
        duplicate_pair_count=duplicate_pair_count,
        weak_support_clusters=weak_support_clusters,
    )


def _build_weak_support_clusters(
    *,
    current_summaries_by_claim: dict[UUID, ClaimRunSummary],
    weak_support_claim_ids: list[UUID],
) -> list[DraftWeakSupportCluster]:
    claims_by_flag: Counter[str] = Counter()
    claim_ids_by_flag: dict[str, list[UUID]] = {}

    for claim_id in weak_support_claim_ids:
        summary = current_summaries_by_claim.get(claim_id)
        if summary is None:
            continue
        flags = summary.deterministic_flags or [StructuredReasoningCategory.WEAK_SUPPORT.value]
        for flag in flags:
            claims_by_flag[flag] += 1
            claim_ids_by_flag.setdefault(flag, []).append(claim_id)

    ranked = sorted(claims_by_flag.items(), key=lambda item: (-item[1], item[0]))
    return [
        DraftWeakSupportCluster(
            flag=flag,
            claim_count=count,
            claim_ids=claim_ids_by_flag.get(flag, []),
        )
        for flag, count in ranked[:5]
    ]


def _pair_count(edges: list[ClaimGraphEdge], relationship_type: ClaimGraphRelationshipType) -> int:
    pair_keys = {
        tuple(sorted((str(edge.source_claim_id), str(edge.target_claim_id))))
        for edge in edges
        if edge.relationship_type == relationship_type
    }
    return len(pair_keys)


def _change_score(summary: ClaimChangeSummary | None) -> int:
    if summary is None or not summary.changed_since_last_run:
        return 0
    score = 0
    score += 3 if summary.verdict_changed else 0
    score += 2 if summary.flags_changed else 0
    score += 2 if summary.reasoning_categories_changed else 0
    score += 1 if summary.support_changed else 0
    score += 1 if summary.confidence_changed else 0
    return score


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _int_value(value: object) -> int:
    return value if isinstance(value, int) else 0


def _float_value(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None
