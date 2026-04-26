"""Read-side claim review history service."""

from uuid import UUID

from sqlalchemy.orm import Session

from app.core.exceptions import NotFoundError
from app.models import ClaimReviewDecision, VerificationRun
from app.repositories.claim_graph_edges import ClaimGraphEdgeRepository
from app.repositories.claims import ClaimsRepository
from app.repositories.draft_review_runs import DraftReviewRunRepository
from app.repositories.review_decisions import ClaimReviewDecisionRepository
from app.repositories.verification import VerificationRepository
from app.schemas.review_decision import ClaimReviewDecisionRead
from app.schemas.review_history import (
    ClaimReviewGraphRelationshipRead,
    ClaimReviewHistoryChangeSummaryRead,
    ClaimReviewHistoryRead,
)
from app.schemas.verification import VerificationRunRead
from app.services.verification.history_read import build_verification_run_history_read
from app.services.workflows.review_intelligence import (
    build_claim_change_summary,
    build_claim_contradiction_flags,
    build_claim_graph_relationship_index,
    build_claim_run_summary_from_verification_run,
)


class ClaimReviewHistoryService:
    """Build an auditable history view for a claim in the review workbench."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.claims = ClaimsRepository(session)
        self.review_runs = DraftReviewRunRepository(session)
        self.claim_graph_edges = ClaimGraphEdgeRepository(session)
        self.review_decisions = ClaimReviewDecisionRepository(session)
        self.verification_runs = VerificationRepository(session)

    def read_claim_history(self, claim_id: UUID) -> ClaimReviewHistoryRead:
        claim = self.claims.get(claim_id)
        if claim is None:
            raise NotFoundError("Claim unit not found.")

        verification_runs = self.verification_runs.list_by_claim(claim_id)
        decision_history = self.review_decisions.list_by_claim(claim_id)
        latest_verification = verification_runs[0] if verification_runs else None
        previous_verification = verification_runs[1] if len(verification_runs) > 1 else None
        latest_decision = decision_history[0] if decision_history else None

        draft_claims = self.claims.list_by_draft(claim.assertion.draft_id)
        latest_review_runs = self.review_runs.list_by_draft(claim.assertion.draft_id, limit=1)
        latest_review_run = latest_review_runs[0] if latest_review_runs else None
        relationship_index = build_claim_graph_relationship_index(
            self.claim_graph_edges.list_by_review_run(latest_review_run.id) if latest_review_run is not None else [],
            claim_text_by_id={draft_claim.id: draft_claim.text for draft_claim in draft_claims},
        )
        claim_relationships = relationship_index.get(claim.id, [])
        contradiction_flags = build_claim_contradiction_flags(
            relationships=claim_relationships,
            verification_runs=verification_runs,
        )
        latest_summary = build_claim_run_summary_from_verification_run(
            claim.id,
            latest_verification,
            fallback_verdict=claim.support_status,
        )
        previous_summary = (
            build_claim_run_summary_from_verification_run(
                claim.id,
                previous_verification,
                fallback_verdict=None,
            )
            if previous_verification is not None
            else None
        )
        change_summary = build_claim_change_summary(latest_summary, previous_summary)

        return ClaimReviewHistoryRead(
            claim_id=claim.id,
            draft_id=claim.assertion.draft_id,
            claim_text=claim.text,
            assertion_context=claim.assertion.raw_text,
            support_status=claim.support_status,
            review_disposition="resolved" if latest_decision is not None else "active",
            latest_decision=_build_decision_read(latest_decision),
            decision_history=[_build_decision_read(decision) for decision in decision_history],
            latest_verification=_build_verification_read(latest_verification),
            previous_verification=_build_verification_read(previous_verification),
            verification_runs=[_build_verification_read(run) for run in verification_runs],
            reasoning_categories=list(latest_summary.reasoning_categories),
            contradiction_flags=contradiction_flags,
            claim_relationships=[
                ClaimReviewGraphRelationshipRead(
                    relationship_type=relationship.relationship_type,
                    related_claim_id=relationship.related_claim_id,
                    related_claim_text=relationship.related_claim_text,
                    reason_code=relationship.reason_code,
                    reason_text=relationship.reason_text,
                    confidence_score=relationship.confidence_score,
                )
                for relationship in claim_relationships
            ],
            change_summary=ClaimReviewHistoryChangeSummaryRead(
                latest_verdict=change_summary.current_verdict,
                previous_verdict=change_summary.previous_verdict,
                verdict_changed=change_summary.verdict_changed,
                latest_confidence_score=change_summary.current_confidence_score,
                previous_confidence_score=change_summary.previous_confidence_score,
                confidence_changed=change_summary.confidence_changed,
                latest_primary_anchor=change_summary.current_primary_anchor,
                previous_primary_anchor=change_summary.previous_primary_anchor,
                primary_anchor_changed=change_summary.primary_anchor_changed,
                latest_flags=list(change_summary.current_flags),
                previous_flags=list(change_summary.previous_flags),
                flags_changed=change_summary.flags_changed,
                latest_reasoning_categories=list(change_summary.current_reasoning_categories),
                previous_reasoning_categories=list(change_summary.previous_reasoning_categories),
                reasoning_categories_changed=change_summary.reasoning_categories_changed,
                latest_support_assessment_count=change_summary.current_support_assessment_count,
                previous_support_assessment_count=change_summary.previous_support_assessment_count,
                latest_excluded_link_count=change_summary.current_excluded_link_count,
                previous_excluded_link_count=change_summary.previous_excluded_link_count,
                support_changed=change_summary.support_changed,
                changed_since_last_run=change_summary.changed_since_last_run,
                latest_decision_at=latest_decision.created_at if latest_decision is not None else None,
                latest_action=latest_decision.action if latest_decision is not None else None,
            ),
        )


def _build_decision_read(decision: ClaimReviewDecision | None) -> ClaimReviewDecisionRead | None:
    if decision is None:
        return None
    return ClaimReviewDecisionRead.model_validate(decision)


def _build_verification_read(run: VerificationRun | None) -> VerificationRunRead | None:
    if run is None:
        return None
    return build_verification_run_history_read(run)
