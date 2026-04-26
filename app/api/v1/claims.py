"""Claim extraction, linking, and review decision routes."""

from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.core.exceptions import NotFoundError
from app.repositories.assertions import AssertionRepository
from app.repositories.claims import ClaimsRepository
from app.repositories.links import LinksRepository
from app.repositories.segments import SegmentRepository
from app.schemas.claim_unit import ClaimUnitRead
from app.schemas.review_history import ClaimReviewHistoryRead
from app.schemas.review_decision import ClaimReviewDecisionCreate, ClaimReviewDecisionMutationRead
from app.schemas.support_link import SupportLinkCreate, SupportLinkRead
from app.services.claims.extractor import ClaimExtractionService
from app.services.workflows.claim_review_history import ClaimReviewHistoryService
from app.services.workflows.review_decisions import ClaimReviewDecisionMutationResult, ClaimReviewDecisionService

router = APIRouter(prefix="/api/v1", tags=["claims"])


@router.post(
    "/assertions/{assertion_id}/extract-claims",
    response_model=list[ClaimUnitRead],
    status_code=status.HTTP_201_CREATED,
)
def extract_claims(assertion_id: UUID, db: Session = Depends(get_db_session)):
    if AssertionRepository(db).get(assertion_id) is None:
        raise NotFoundError("Assertion not found.")

    claims = ClaimExtractionService(db).extract_from_assertion(assertion_id)
    db.commit()
    for claim in claims:
        db.refresh(claim)
    return claims


@router.get("/assertions/{assertion_id}/claims", response_model=list[ClaimUnitRead])
def list_claims(assertion_id: UUID, db: Session = Depends(get_db_session)):
    if AssertionRepository(db).get(assertion_id) is None:
        raise NotFoundError("Assertion not found.")
    return ClaimsRepository(db).list_by_assertion(assertion_id)


@router.post("/claims/{claim_id}/links", response_model=SupportLinkRead, status_code=status.HTTP_201_CREATED)
def create_link(
    claim_id: UUID,
    payload: SupportLinkCreate,
    db: Session = Depends(get_db_session),
):
    if ClaimsRepository(db).get(claim_id) is None:
        raise NotFoundError("Claim unit not found.")
    if SegmentRepository(db).get(payload.segment_id) is None:
        raise NotFoundError("Segment not found.")

    link = LinksRepository(db).create(claim_id, payload)
    db.commit()
    db.refresh(link)
    return link


@router.get("/claims/{claim_id}/links", response_model=list[SupportLinkRead])
def list_links(claim_id: UUID, db: Session = Depends(get_db_session)):
    if ClaimsRepository(db).get(claim_id) is None:
        raise NotFoundError("Claim unit not found.")
    return LinksRepository(db).list_by_claim(claim_id)


@router.get("/claims/{claim_id}/review-history", response_model=ClaimReviewHistoryRead)
def get_claim_review_history(claim_id: UUID, db: Session = Depends(get_db_session)):
    return ClaimReviewHistoryService(db).read_claim_history(claim_id)


@router.post(
    "/claims/{claim_id}/decisions",
    response_model=ClaimReviewDecisionMutationRead,
    status_code=status.HTTP_201_CREATED,
)
def create_claim_review_decision(
    claim_id: UUID,
    payload: ClaimReviewDecisionCreate,
    db: Session = Depends(get_db_session),
):
    result = ClaimReviewDecisionService(db).record_decision(claim_id, payload)
    db.commit()
    db.refresh(result.decision)
    return _build_claim_review_decision_response(result)


def _build_claim_review_decision_response(
    result: ClaimReviewDecisionMutationResult,
) -> dict[str, object]:
    return {
        "decision": {
            "id": result.decision.id,
            "claim_unit_id": result.decision.claim_unit_id,
            "draft_id": result.decision.draft_id,
            "verification_run_id": result.decision.verification_run_id,
            "action": result.decision.action,
            "note": result.decision.note,
            "proposed_replacement_text": result.decision.proposed_replacement_text,
            "created_at": result.decision.created_at,
        },
        "claim_review_state": {
            "claim_id": result.claim_review_state.claim_id,
            "draft_id": result.claim_review_state.draft_id,
            "review_status": result.claim_review_state.review_status,
            "latest_action": result.claim_review_state.latest_action,
            "decision_count": result.claim_review_state.decision_count,
            "latest_verification_run_id": result.claim_review_state.latest_verification_run_id,
            "latest_verdict": result.claim_review_state.latest_verdict,
            "removed_from_active_queue": result.claim_review_state.removed_from_active_queue,
        },
        "draft_queue": {
            "draft_id": result.draft_queue.draft_id,
            "total_flagged_claims": result.draft_queue.total_flagged_claims,
            "resolved_flagged_claims": result.draft_queue.resolved_flagged_claims,
            "remaining_flagged_claims": result.draft_queue.remaining_flagged_claims,
            "next_claim_id": result.draft_queue.next_claim_id,
        },
    }
