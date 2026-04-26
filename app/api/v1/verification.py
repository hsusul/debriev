"""Verification and decision routes."""

from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.core.exceptions import NotFoundError
from app.repositories.claims import ClaimsRepository
from app.repositories.decisions import DecisionsRepository
from app.repositories.verification import VerificationRepository
from app.schemas.verification import (
    SupportAssessmentRead,
    UserDecisionCreate,
    UserDecisionRead,
    VerificationRequest,
    VerificationResultRead,
    VerificationRunRead,
)
from app.services.verification.history_read import build_verification_run_history_read
from app.services.verification.classifier import ClaimVerificationService, VerificationExecution

router = APIRouter(prefix="/api/v1", tags=["verification"])


@router.post("/claims/{claim_id}/verify", response_model=VerificationResultRead, status_code=status.HTTP_201_CREATED)
def verify_claim(
    claim_id: UUID,
    payload: VerificationRequest | None = None,
    db: Session = Depends(get_db_session),
):
    execution = ClaimVerificationService(db).verify_claim(
        claim_id,
        model_version=payload.model_version if payload else None,
        prompt_version=payload.prompt_version if payload else None,
    )
    return _build_verification_result_response(execution)


@router.get("/claims/{claim_id}/verification-runs", response_model=list[VerificationRunRead])
def list_verification_runs(claim_id: UUID, db: Session = Depends(get_db_session)):
    if ClaimsRepository(db).get(claim_id) is None:
        raise NotFoundError("Claim unit not found.")
    return [
        _build_verification_run_history_response(run)
        for run in VerificationRepository(db).list_by_claim(claim_id)
    ]


@router.post(
    "/verification-runs/{run_id}/decisions",
    response_model=UserDecisionRead,
    status_code=status.HTTP_201_CREATED,
)
def create_decision(
    run_id: UUID,
    payload: UserDecisionCreate,
    db: Session = Depends(get_db_session),
):
    if VerificationRepository(db).get(run_id) is None:
        raise NotFoundError("Verification run not found.")

    decision = DecisionsRepository(db).create(run_id, payload)
    db.commit()
    db.refresh(decision)
    return decision


def _build_verification_result_response(execution: VerificationExecution) -> dict[str, object]:
    run = execution.run
    result = execution.result
    return {
        "id": run.id,
        "claim_unit_id": run.claim_unit_id,
        "model_version": run.model_version,
        "prompt_version": run.prompt_version,
        "deterministic_flags": run.deterministic_flags,
        "reasoning_categories": run.reasoning_categories,
        "verdict": run.verdict,
        "reasoning": run.reasoning,
        "suggested_fix": run.suggested_fix,
        "confidence_score": run.confidence_score,
        "created_at": run.created_at,
        "primary_anchor": result.primary_anchor,
        "support_assessments": [
            SupportAssessmentRead(
                segment_id=assessment.segment_id,
                anchor=assessment.anchor,
                role=assessment.role,
                contribution=assessment.contribution,
            ).model_dump()
            for assessment in result.support_assessments
        ],
    }


def _build_verification_run_history_response(run) -> dict[str, object]:
    return build_verification_run_history_read(run).model_dump(mode="json")
