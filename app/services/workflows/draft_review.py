"""Draft review workflow layered on top of draft compile output."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.enums import ClaimReviewAction, DraftReviewRunStatus, SupportStatus
from app.models import (
    CURRENT_DRAFT_REVIEW_RUN_SNAPSHOT_VERSION,
    ClaimGraphEdge,
    ClaimReviewDecision,
    ClaimUnit,
    DraftReviewRun,
    VerificationRun,
)
from app.core.exceptions import NotFoundError
from app.repositories.claims import ClaimsRepository
from app.repositories.claim_graph_edges import ClaimGraphEdgeRepository
from app.repositories.draft_review_runs import DraftReviewRunRepository
from app.repositories.drafts import DraftRepository
from app.repositories.review_decisions import ClaimReviewDecisionRepository
from app.services.audit.report_builder import build_draft_review_summary
from app.services.llm.base import ProviderSupportAssessment
from app.services.workflows.claim_graph import build_claim_graph_edges
from app.services.workflows.draft_compile import (
    FLAGGED_VERDICTS,
    DraftCompileFlaggedClaim,
    DraftCompileResult,
    DraftCompileService,
    DraftCompileVerdictCounts,
)
from app.services.workflows.review_intelligence import (
    ClaimChangeSummary,
    DraftReviewIntelligenceSummary,
    build_claim_change_summary,
    build_claim_contradiction_flags,
    build_claim_graph_relationship_index,
    build_claim_run_summary_from_snapshot_entry,
    build_claim_run_summary_from_verification_run,
    build_draft_review_intelligence_summary,
)
from app.services.workflows.review_queue import project_review_queue
from app.services.verification.snapshot_adapter import parse_verification_support_snapshot

RISK_PRIORITY = {
    SupportStatus.UNSUPPORTED: 0,
    SupportStatus.OVERSTATED: 1,
    SupportStatus.AMBIGUOUS: 2,
    SupportStatus.UNVERIFIED: 3,
}
VERDICT_BUCKET_ORDER = (
    SupportStatus.UNSUPPORTED,
    SupportStatus.OVERSTATED,
    SupportStatus.AMBIGUOUS,
    SupportStatus.UNVERIFIED,
)
ReviewStateSource = Literal["fresh_execution", "persisted_read"]


@dataclass(slots=True)
class DraftReviewFlaggedClaimCounts:
    unsupported: int = 0
    overstated: int = 0
    ambiguous: int = 0
    unverified: int = 0

    @property
    def total(self) -> int:
        return self.unsupported + self.ambiguous + self.overstated + self.unverified


@dataclass(slots=True)
class DraftReviewIssueBuckets:
    unsupported: list[DraftCompileFlaggedClaim] = field(default_factory=list)
    overstated: list[DraftCompileFlaggedClaim] = field(default_factory=list)
    ambiguous: list[DraftCompileFlaggedClaim] = field(default_factory=list)
    unverified: list[DraftCompileFlaggedClaim] = field(default_factory=list)


@dataclass(slots=True)
class DraftReviewFlagBucket:
    flag: str
    claim_count: int
    claims: list[DraftCompileFlaggedClaim] = field(default_factory=list)


@dataclass(slots=True)
class DraftReviewOverview:
    total_claims: int
    total_flagged_claims: int
    highest_severity_bucket: SupportStatus | None
    top_issue_categories: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DraftReviewQueueState:
    draft_id: UUID
    total_flagged_claims: int
    resolved_flagged_claims: int
    remaining_flagged_claims: int
    next_claim_id: UUID | None


@dataclass(slots=True)
class DraftReviewFreshness:
    state_source: ReviewStateSource
    has_persisted_review_runs: bool
    last_review_run_at: datetime | None
    latest_review_run_id: UUID | None
    latest_review_run_status: DraftReviewRunStatus | None
    latest_decision_at: datetime | None
    has_decisions_after_latest_run: bool
    latest_claim_verification_at: datetime | None
    latest_verification_run_id: UUID | None
    has_verification_activity_after_latest_run: bool
    is_stale: bool


@dataclass(slots=True)
class DraftReviewRunSummary:
    run_id: UUID
    status: DraftReviewRunStatus
    created_at: datetime
    total_claims: int
    total_flagged_claims: int
    resolved_flagged_claims: int
    remaining_flagged_claims: int
    highest_severity_bucket: SupportStatus | None


@dataclass(slots=True)
class DraftReviewDecisionSummary:
    action: ClaimReviewAction
    note: str | None
    proposed_replacement_text: str | None
    created_at: datetime


@dataclass(slots=True)
class DraftReviewResolvedClaim:
    claim: DraftCompileFlaggedClaim
    latest_decision: DraftReviewDecisionSummary


@dataclass(slots=True)
class DraftReviewResult:
    draft_id: UUID
    total_claims: int
    verdict_counts: DraftCompileVerdictCounts
    flagged_claim_counts: DraftReviewFlaggedClaimCounts
    review_overview: DraftReviewOverview
    freshness: DraftReviewFreshness
    queue_state: DraftReviewQueueState
    issue_buckets: DraftReviewIssueBuckets
    active_queue_claims: list[DraftCompileFlaggedClaim] = field(default_factory=list)
    resolved_claims: list[DraftReviewResolvedClaim] = field(default_factory=list)
    latest_review_run: DraftReviewRunSummary | None = None
    previous_review_run: DraftReviewRunSummary | None = None
    intelligence_summary: DraftReviewIntelligenceSummary | None = None
    flag_buckets: list[DraftReviewFlagBucket] = field(default_factory=list)
    top_risky_claims: list[DraftCompileFlaggedClaim] = field(default_factory=list)
    summary: str = ""


class DraftReviewService:
    """Convert compile output into reviewer-friendly issue groupings and ranking."""

    def review_draft(
        self,
        compile_result: DraftCompileResult,
        *,
        latest_decisions_by_claim: dict[UUID, ClaimReviewDecision] | None = None,
        freshness: DraftReviewFreshness | None = None,
        top_claim_limit: int = 5,
    ) -> DraftReviewResult:
        latest_decisions_by_claim = latest_decisions_by_claim or {}
        queue_projection = project_review_queue(
            [claim.claim_id for claim in compile_result.flagged_claims],
            latest_decisions_by_claim=latest_decisions_by_claim,
        )
        active_claim_id_set = set(queue_projection.active_claim_ids)
        resolved_claim_id_set = set(queue_projection.resolved_claim_ids)
        active_flagged_claims = [
            claim for claim in compile_result.flagged_claims if claim.claim_id in active_claim_id_set
        ]
        resolved_claims = [
            DraftReviewResolvedClaim(
                claim=claim,
                latest_decision=_build_review_decision_summary(latest_decisions_by_claim[claim.claim_id]),
            )
            for claim in compile_result.flagged_claims
            if claim.claim_id in resolved_claim_id_set
        ]

        issue_buckets = _group_flagged_claims_by_verdict(active_flagged_claims)
        flagged_claim_counts = DraftReviewFlaggedClaimCounts(
            unsupported=len(issue_buckets.unsupported),
            overstated=len(issue_buckets.overstated),
            ambiguous=len(issue_buckets.ambiguous),
            unverified=len(issue_buckets.unverified),
        )
        flag_buckets = _group_flagged_claims_by_flag(active_flagged_claims)
        result = DraftReviewResult(
            draft_id=compile_result.draft_id,
            total_claims=compile_result.total_claims,
            verdict_counts=_copy_verdict_counts(compile_result.counts),
            flagged_claim_counts=flagged_claim_counts,
            review_overview=_build_review_overview(
                total_claims=compile_result.total_claims,
                flagged_claim_counts=flagged_claim_counts,
                issue_buckets=issue_buckets,
                flag_buckets=flag_buckets,
            ),
            freshness=freshness
            or DraftReviewFreshness(
                state_source="persisted_read",
                has_persisted_review_runs=False,
                last_review_run_at=None,
                latest_review_run_id=None,
                latest_review_run_status=None,
                latest_decision_at=None,
                has_decisions_after_latest_run=False,
                latest_claim_verification_at=None,
                latest_verification_run_id=None,
                has_verification_activity_after_latest_run=False,
                is_stale=True,
            ),
            queue_state=DraftReviewQueueState(
                draft_id=compile_result.draft_id,
                total_flagged_claims=queue_projection.total_flagged_claims,
                resolved_flagged_claims=queue_projection.resolved_flagged_claims,
                remaining_flagged_claims=queue_projection.remaining_flagged_claims,
                next_claim_id=queue_projection.next_claim_id,
            ),
            active_queue_claims=active_flagged_claims,
            resolved_claims=resolved_claims,
            issue_buckets=issue_buckets,
            flag_buckets=flag_buckets,
            top_risky_claims=_rank_top_risky_claims(active_flagged_claims, limit=top_claim_limit),
        )
        result.summary = build_draft_review_summary(result)
        return result


class DraftReviewReadService:
    """Read current review queue state from persisted claim and verification data."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.drafts = DraftRepository(session)
        self.claims = ClaimsRepository(session)
        self.review_decisions = ClaimReviewDecisionRepository(session)
        self.review_runs = DraftReviewRunRepository(session)
        self.claim_graph_edges = ClaimGraphEdgeRepository(session)

    def read_review_state(
        self,
        draft_id: UUID,
        *,
        state_source: ReviewStateSource = "persisted_read",
        top_claim_limit: int = 5,
    ) -> DraftReviewResult:
        if self.drafts.get(draft_id) is None:
            raise NotFoundError("Draft not found.")

        draft_claims = self.claims.list_by_draft(draft_id)
        draft_review_runs = self.review_runs.list_by_draft(draft_id, limit=2)
        latest_review_run = draft_review_runs[0] if draft_review_runs else None
        previous_review_run = draft_review_runs[1] if len(draft_review_runs) > 1 else None
        draft_decisions = self.review_decisions.list_by_draft(draft_id)
        compile_result = _build_compile_result_from_persisted_claims(draft_id, draft_claims)
        result = DraftReviewService().review_draft(
            compile_result,
            latest_decisions_by_claim=_latest_review_decisions_by_claim(draft_decisions),
            freshness=build_review_freshness(
                draft_claims,
                latest_review_run=latest_review_run,
                latest_decision=draft_decisions[0] if draft_decisions else None,
                state_source=state_source,
            ),
            top_claim_limit=top_claim_limit,
        )
        _enrich_review_result(
            review_result=result,
            draft_claims=draft_claims,
            previous_review_run=previous_review_run,
            claim_graph_edges=(
                self.claim_graph_edges.list_by_review_run(latest_review_run.id)
                if latest_review_run is not None
                else []
            ),
        )
        result.latest_review_run = _build_review_run_summary(latest_review_run)
        result.previous_review_run = _build_review_run_summary(previous_review_run)
        return result


class DraftReviewExecutionService:
    """Execute a fresh draft review and persist a draft-level execution record."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.claims = ClaimsRepository(session)
        self.review_decisions = ClaimReviewDecisionRepository(session)
        self.review_runs = DraftReviewRunRepository(session)
        self.claim_graph_edges = ClaimGraphEdgeRepository(session)

    def execute_review(
        self,
        draft_id: UUID,
        *,
        top_claim_limit: int = 5,
        model_version: str | None = None,
        prompt_version: str | None = None,
    ) -> DraftReviewResult:
        compile_result = DraftCompileService(self.session).compile_draft(
            draft_id,
            model_version=model_version,
            prompt_version=prompt_version,
        )
        latest_decisions_by_claim = self.review_decisions.latest_by_draft(draft_id)
        review_result = DraftReviewService().review_draft(
            compile_result,
            latest_decisions_by_claim=latest_decisions_by_claim,
            top_claim_limit=top_claim_limit,
        )
        draft_claims = self.claims.list_by_draft(draft_id)
        review_run = self.review_runs.create(
            draft_id=draft_id,
            status=DraftReviewRunStatus.COMPLETED,
            total_claims=review_result.total_claims,
            total_flagged_claims=review_result.flagged_claim_counts.total,
            resolved_flagged_claims=review_result.queue_state.resolved_flagged_claims,
            remaining_flagged_claims=review_result.queue_state.remaining_flagged_claims,
            highest_severity_bucket=(
                review_result.review_overview.highest_severity_bucket.value
                if review_result.review_overview.highest_severity_bucket is not None
                else None
            ),
            snapshot_version=CURRENT_DRAFT_REVIEW_RUN_SNAPSHOT_VERSION,
            snapshot=_build_review_run_snapshot(review_result, draft_claims),
        )
        graph_summary = build_claim_graph_edges(
            draft_id=draft_id,
            draft_review_run_id=review_run.id,
            claims=draft_claims,
        )
        self.claim_graph_edges.create_many(graph_summary.edges)
        return DraftReviewReadService(self.session).read_review_state(
            draft_id,
            state_source="fresh_execution",
            top_claim_limit=top_claim_limit,
        )


def build_review_freshness(
    claims: list[ClaimUnit],
    *,
    latest_review_run: DraftReviewRun | None,
    latest_decision: ClaimReviewDecision | None,
    state_source: ReviewStateSource,
) -> DraftReviewFreshness:
    latest_run: VerificationRun | None = None
    for claim in claims:
        claim_latest_run = _latest_verification_run(getattr(claim, "verification_runs", []))
        if claim_latest_run is None:
            continue
        if latest_run is None or (claim_latest_run.created_at, str(claim_latest_run.id)) > (
            latest_run.created_at,
            str(latest_run.id),
        ):
            latest_run = claim_latest_run

    has_persisted_review_runs = latest_review_run is not None
    has_decisions_after_latest_run = _is_created_after(latest_decision, latest_review_run)
    has_verification_activity_after_latest_run = _is_created_after(latest_run, latest_review_run)
    return DraftReviewFreshness(
        state_source=state_source,
        has_persisted_review_runs=has_persisted_review_runs,
        last_review_run_at=latest_review_run.created_at if latest_review_run is not None else None,
        latest_review_run_id=latest_review_run.id if latest_review_run is not None else None,
        latest_review_run_status=latest_review_run.status if latest_review_run is not None else None,
        latest_decision_at=latest_decision.created_at if latest_decision is not None else None,
        has_decisions_after_latest_run=has_decisions_after_latest_run,
        latest_claim_verification_at=latest_run.created_at if latest_run is not None else None,
        latest_verification_run_id=latest_run.id if latest_run is not None else None,
        has_verification_activity_after_latest_run=has_verification_activity_after_latest_run,
        is_stale=(
            latest_review_run is None
            or has_decisions_after_latest_run
            or has_verification_activity_after_latest_run
        ),
    )


def _latest_review_decisions_by_claim(
    decisions: list[ClaimReviewDecision],
) -> dict[UUID, ClaimReviewDecision]:
    latest_by_claim: dict[UUID, ClaimReviewDecision] = {}
    for decision in decisions:
        if decision.claim_unit_id not in latest_by_claim:
            latest_by_claim[decision.claim_unit_id] = decision
    return latest_by_claim


def _build_review_run_summary(run: DraftReviewRun | None) -> DraftReviewRunSummary | None:
    if run is None:
        return None
    return DraftReviewRunSummary(
        run_id=run.id,
        status=run.status,
        created_at=run.created_at,
        total_claims=run.total_claims,
        total_flagged_claims=run.total_flagged_claims,
        resolved_flagged_claims=run.resolved_flagged_claims,
        remaining_flagged_claims=run.remaining_flagged_claims,
        highest_severity_bucket=(
            SupportStatus(run.highest_severity_bucket)
            if run.highest_severity_bucket is not None
            else None
        ),
    )


def _build_review_run_snapshot(result: DraftReviewResult, draft_claims: list[ClaimUnit]) -> dict[str, object]:
    claim_summaries = [
        _build_claim_snapshot_summary(claim)
        for claim in draft_claims
    ]
    return {
        "summary": {
            "total_claims": result.total_claims,
            "flagged_claim_counts": {
                "unsupported": result.flagged_claim_counts.unsupported,
                "overstated": result.flagged_claim_counts.overstated,
                "ambiguous": result.flagged_claim_counts.ambiguous,
                "unverified": result.flagged_claim_counts.unverified,
                "total": result.flagged_claim_counts.total,
            },
            "review_overview": {
                "highest_severity_bucket": (
                    result.review_overview.highest_severity_bucket.value
                    if result.review_overview.highest_severity_bucket is not None
                    else None
                ),
                "top_issue_categories": list(result.review_overview.top_issue_categories),
            },
            "queue_state": {
                "total_flagged_claims": result.queue_state.total_flagged_claims,
                "resolved_flagged_claims": result.queue_state.resolved_flagged_claims,
                "remaining_flagged_claims": result.queue_state.remaining_flagged_claims,
            },
        },
        "claims": claim_summaries,
    }


def _build_claim_snapshot_summary(claim: ClaimUnit) -> dict[str, object]:
    latest_run = _latest_verification_run(getattr(claim, "verification_runs", []))
    summary = build_claim_run_summary_from_verification_run(
        claim.id,
        latest_run,
        fallback_verdict=claim.support_status,
    )
    return {
        "claim_id": str(claim.id),
        "verdict": summary.verdict.value if summary.verdict is not None else None,
        "confidence_score": summary.confidence_score,
        "deterministic_flags": list(summary.deterministic_flags),
        "reasoning_categories": list(summary.reasoning_categories),
        "primary_anchor": summary.primary_anchor,
        "support_assessment_count": summary.support_assessment_count,
        "excluded_link_count": summary.excluded_link_count,
        "latest_verification_run_id": (
            str(summary.latest_verification_run_id) if summary.latest_verification_run_id is not None else None
        ),
    }


def _parse_claim_summaries_from_review_run(run: DraftReviewRun | None) -> dict[UUID, object]:
    if run is None:
        return {}
    snapshot = run.snapshot if isinstance(run.snapshot, dict) else {}
    claims = snapshot.get("claims")
    if not isinstance(claims, list):
        return {}
    summaries: dict[UUID, object] = {}
    for entry in claims:
        parsed = build_claim_run_summary_from_snapshot_entry(entry)
        if parsed is None:
            continue
        summaries[parsed.claim_id] = parsed
    return summaries


def _enrich_review_result(
    *,
    review_result: DraftReviewResult,
    draft_claims: list[ClaimUnit],
    previous_review_run: DraftReviewRun | None,
    claim_graph_edges: list[ClaimGraphEdge],
) -> None:
    claim_text_by_id = {claim.id: claim.text for claim in draft_claims}
    current_summaries_by_claim = {
        claim.id: build_claim_run_summary_from_verification_run(
            claim.id,
            _latest_verification_run(getattr(claim, "verification_runs", [])),
            fallback_verdict=claim.support_status,
        )
        for claim in draft_claims
    }
    previous_summaries_by_claim = _parse_claim_summaries_from_review_run(previous_review_run)
    relationship_index = build_claim_graph_relationship_index(
        claim_graph_edges,
        claim_text_by_id=claim_text_by_id,
    )
    contradiction_flags_by_claim = {
        claim.id: build_claim_contradiction_flags(
            relationships=relationship_index.get(claim.id, []),
            verification_runs=getattr(claim, "verification_runs", []),
        )
        for claim in draft_claims
    }
    change_summaries_by_claim = {
        claim_id: build_claim_change_summary(summary, previous_summaries_by_claim.get(claim_id))
        for claim_id, summary in current_summaries_by_claim.items()
    }

    flagged_claims = {
        claim.claim_id: claim
        for claim in review_result.active_queue_claims
    }
    flagged_claims.update({record.claim.claim_id: record.claim for record in review_result.resolved_claims})
    for claim_id, flagged_claim in flagged_claims.items():
        current_summary = current_summaries_by_claim.get(claim_id)
        change_summary = change_summaries_by_claim.get(claim_id)
        if current_summary is not None:
            flagged_claim.reasoning_categories = list(current_summary.reasoning_categories)
        flagged_claim.change_summary = change_summary
        flagged_claim.changed_since_last_run = change_summary.changed_since_last_run if change_summary else False
        flagged_claim.contradiction_flags = list(contradiction_flags_by_claim.get(claim_id, []))
        flagged_claim.claim_relationships = [
            DraftCompileFlaggedClaim.Relationship(
                relationship_type=relationship.relationship_type,
                related_claim_id=relationship.related_claim_id,
                related_claim_text=relationship.related_claim_text,
                reason_code=relationship.reason_code,
                reason_text=relationship.reason_text,
                confidence_score=relationship.confidence_score,
            )
            for relationship in relationship_index.get(claim_id, [])
        ]

    review_result.intelligence_summary = build_draft_review_intelligence_summary(
        claims=draft_claims,
        current_summaries_by_claim=current_summaries_by_claim,
        change_summaries_by_claim=change_summaries_by_claim,
        contradiction_flags_by_claim=contradiction_flags_by_claim,
        relationship_index=relationship_index,
        edges=claim_graph_edges,
        risk_distribution=_build_risk_distribution(review_result.verdict_counts),
    )


def _build_review_decision_summary(decision: ClaimReviewDecision) -> DraftReviewDecisionSummary:
    return DraftReviewDecisionSummary(
        action=decision.action,
        note=decision.note,
        proposed_replacement_text=decision.proposed_replacement_text,
        created_at=decision.created_at,
    )


def _build_compile_result_from_persisted_claims(
    draft_id: UUID,
    claims: list[ClaimUnit],
) -> DraftCompileResult:
    counts = DraftCompileVerdictCounts()
    flagged_claims: list[DraftCompileFlaggedClaim] = []

    for draft_sequence, claim in enumerate(claims, start=1):
        latest_run = _latest_verification_run(getattr(claim, "verification_runs", []))
        verdict = latest_run.verdict if latest_run is not None else claim.support_status
        counts.increment(verdict)
        if verdict not in FLAGGED_VERDICTS:
            continue
        flagged_claims.append(
            _build_persisted_flagged_claim(
                claim,
                draft_sequence=draft_sequence,
                verdict=verdict,
                latest_run=latest_run,
            )
        )

    return DraftCompileResult(
        draft_id=draft_id,
        total_claims=len(claims),
        counts=counts,
        flagged_claims=flagged_claims,
    )


def _build_persisted_flagged_claim(
    claim: object,
    *,
    draft_sequence: int,
    verdict: SupportStatus,
    latest_run: object | None,
) -> DraftCompileFlaggedClaim:
    primary_anchor = None
    support_assessments: list[ProviderSupportAssessment] = []
    suggested_fix = None
    confidence_score = None
    deterministic_flags: list[str] = []
    reasoning_categories: list[str] = []

    if latest_run is not None:
        deterministic_flags = list(latest_run.deterministic_flags)
        reasoning_categories = list(latest_run.reasoning_categories)
        reasoning = latest_run.reasoning
        suggested_fix = latest_run.suggested_fix
        confidence_score = latest_run.confidence_score
        parsed_snapshot = parse_verification_support_snapshot(
            latest_run.support_snapshot,
            latest_run.support_snapshot_version,
        )
        primary_anchor = parsed_snapshot.provider_output.primary_anchor
        support_assessments = _build_persisted_support_assessments(parsed_snapshot)
        excluded_links = [
            {
                "code": link.code,
                "message": link.message,
            }
            for link in parsed_snapshot.excluded_support_links
        ]
        scope = None
        if parsed_snapshot.claim_scope is not None:
            scope = {
                "scope_kind": parsed_snapshot.claim_scope.scope_kind,
                "allowed_source_document_count": len(parsed_snapshot.claim_scope.allowed_source_document_ids),
            }
    else:
        reasoning = None
        excluded_links = []
        scope = None

    return DraftCompileFlaggedClaim(
        claim_id=claim.id,
        claim_text=claim.text,
        verdict=verdict,
        deterministic_flags=deterministic_flags,
        primary_anchor=primary_anchor,
        draft_sequence=draft_sequence,
        assertion_context=claim.assertion.raw_text,
        reasoning=reasoning,
        excluded_links=excluded_links,
        scope=scope,
        latest_verification_run_id=latest_run.id if latest_run is not None else None,
        latest_verification_run_at=latest_run.created_at if latest_run is not None else None,
        support_assessments=support_assessments,
        suggested_fix=suggested_fix,
        confidence_score=confidence_score,
        reasoning_categories=reasoning_categories,
    )


def _build_persisted_support_assessments(parsed_snapshot) -> list[ProviderSupportAssessment]:
    assessments: list[ProviderSupportAssessment] = []
    for assessment in parsed_snapshot.provider_output.support_assessments:
        segment_id = _coerce_uuid(assessment.segment_id)
        if segment_id is None:
            continue
        assessments.append(
            ProviderSupportAssessment(
                segment_id=segment_id,
                anchor=assessment.anchor,
                role=assessment.role,
                contribution=assessment.contribution,
            )
        )
    return assessments


def _coerce_uuid(value: str | None) -> UUID | None:
    if value is None:
        return None
    try:
        return UUID(value)
    except (TypeError, ValueError):
        return None


def _latest_verification_run(runs: list[VerificationRun]) -> VerificationRun | None:
    if not runs:
        return None
    return max(runs, key=lambda run: (run.created_at, str(run.id)))


def _is_created_after(current: object | None, reference: object | None) -> bool:
    if current is None or reference is None:
        return False
    current_created_at = getattr(current, "created_at", None)
    reference_created_at = getattr(reference, "created_at", None)
    if current_created_at is None or reference_created_at is None:
        return False
    if getattr(current_created_at, "tzinfo", None) is None:
        current_created_at = current_created_at.replace(tzinfo=timezone.utc)
    if getattr(reference_created_at, "tzinfo", None) is None:
        reference_created_at = reference_created_at.replace(tzinfo=timezone.utc)
    return current_created_at > reference_created_at


def _copy_verdict_counts(counts: DraftCompileVerdictCounts) -> DraftCompileVerdictCounts:
    return DraftCompileVerdictCounts(
        supported=counts.supported,
        partially_supported=counts.partially_supported,
        overstated=counts.overstated,
        ambiguous=counts.ambiguous,
        unsupported=counts.unsupported,
        unverified=counts.unverified,
    )


def _group_flagged_claims_by_verdict(flagged_claims: list[DraftCompileFlaggedClaim]) -> DraftReviewIssueBuckets:
    buckets = DraftReviewIssueBuckets()
    for claim in flagged_claims:
        if claim.verdict == SupportStatus.UNSUPPORTED:
            buckets.unsupported.append(claim)
        elif claim.verdict == SupportStatus.AMBIGUOUS:
            buckets.ambiguous.append(claim)
        elif claim.verdict == SupportStatus.OVERSTATED:
            buckets.overstated.append(claim)
        elif claim.verdict == SupportStatus.UNVERIFIED:
            buckets.unverified.append(claim)
    return buckets


def _group_flagged_claims_by_flag(flagged_claims: list[DraftCompileFlaggedClaim]) -> list[DraftReviewFlagBucket]:
    buckets: dict[str, list[DraftCompileFlaggedClaim]] = {}
    for claim in flagged_claims:
        for flag in claim.deterministic_flags:
            buckets.setdefault(flag, []).append(claim)
    ranked_buckets = sorted(
        buckets.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    return [
        DraftReviewFlagBucket(
            flag=flag,
            claim_count=len(claims),
            claims=claims,
        )
        for flag, claims in ranked_buckets
    ]


def _build_review_overview(
    *,
    total_claims: int,
    flagged_claim_counts: DraftReviewFlaggedClaimCounts,
    issue_buckets: DraftReviewIssueBuckets,
    flag_buckets: list[DraftReviewFlagBucket],
) -> DraftReviewOverview:
    return DraftReviewOverview(
        total_claims=total_claims,
        total_flagged_claims=flagged_claim_counts.total,
        highest_severity_bucket=_highest_severity_bucket(issue_buckets),
        top_issue_categories=[bucket.flag for bucket in flag_buckets[:3]],
    )


def _highest_severity_bucket(issue_buckets: DraftReviewIssueBuckets) -> SupportStatus | None:
    for verdict in VERDICT_BUCKET_ORDER:
        if verdict == SupportStatus.UNSUPPORTED and issue_buckets.unsupported:
            return verdict
        if verdict == SupportStatus.OVERSTATED and issue_buckets.overstated:
            return verdict
        if verdict == SupportStatus.AMBIGUOUS and issue_buckets.ambiguous:
            return verdict
        if verdict == SupportStatus.UNVERIFIED and issue_buckets.unverified:
            return verdict
    return None


def _rank_top_risky_claims(
    flagged_claims: list[DraftCompileFlaggedClaim],
    *,
    limit: int,
) -> list[DraftCompileFlaggedClaim]:
    ranked = sorted(
        enumerate(flagged_claims),
        key=lambda item: (
            RISK_PRIORITY.get(item[1].verdict, 99),
            -len(item[1].deterministic_flags),
            _confidence_sort_value(item[1].confidence_score),
            item[0],
        ),
    )
    return [claim for _, claim in ranked[:limit]]


def _confidence_sort_value(confidence: float | None) -> float:
    if confidence is None:
        return -1.0
    return confidence


def _build_risk_distribution(counts: DraftCompileVerdictCounts) -> dict[str, int]:
    return {
        "supported": counts.supported,
        "partially_supported": counts.partially_supported,
        "overstated": counts.overstated,
        "ambiguous": counts.ambiguous,
        "unsupported": counts.unsupported,
        "unverified": counts.unverified,
    }
