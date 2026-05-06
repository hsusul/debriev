"""Verification and decision routes."""

from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.core.exceptions import NotFoundError
from app.repositories.claims import ClaimsRepository
from app.repositories.decisions import DecisionsRepository
from app.repositories.verification import VerificationRepository
from app.schemas.case_pdf_verification import (
    CasePdfAuthorityMetadataRead,
    CasePdfVerificationResultRead,
)
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
from app.services.workflows.case_pdf_verification import CasePdfVerificationService

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


@router.post(
    "/case-pdf-verification",
    response_model=CasePdfVerificationResultRead,
    status_code=status.HTTP_201_CREATED,
)
async def verify_case_pdf(
    pdf_file: UploadFile = File(...),
    statement_text: str = Form(...),
    citation_text: str | None = Form(default=None),
    db: Session = Depends(get_db_session),
):
    result = CasePdfVerificationService().verify_pdf(
        filename=pdf_file.filename,
        pdf_bytes=await pdf_file.read(),
        statement_text=statement_text,
        citation_text=citation_text,
    )
    return _build_case_pdf_verification_response(result)


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


def _build_case_pdf_verification_response(result) -> CasePdfVerificationResultRead:
    return CasePdfVerificationResultRead(
        pdf_text_status=result.pdf_text_status,
        extracted_authority_metadata=(
            CasePdfAuthorityMetadataRead(
                case_name=result.extracted_authority_metadata.case_name,
                reporter_volume=result.extracted_authority_metadata.reporter_volume,
                reporter_abbreviation=result.extracted_authority_metadata.reporter_abbreviation,
                first_page=result.extracted_authority_metadata.first_page,
                court=result.extracted_authority_metadata.court,
                year=result.extracted_authority_metadata.year,
                canonical_citation=result.extracted_authority_metadata.canonical_citation,
            )
            if result.extracted_authority_metadata is not None
            else None
        ),
        extracted_character_count=result.extracted_character_count,
        page_count=result.page_count,
        extraction_warnings=result.extraction_warnings,
        extracted_text_preview=result.extracted_text_preview,
        citation_match_status=result.citation_match_status,
        statement_verdict=result.statement_verdict,
        reasoning=result.reasoning,
        support_snippet=result.support_snippet,
        confidence_score=result.confidence_score,
        suggested_fix=result.suggested_fix,
    )
