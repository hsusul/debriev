"""Assertion routes."""

from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.core.exceptions import NotFoundError
from app.repositories.assertions import AssertionRepository
from app.repositories.drafts import DraftRepository
from app.schemas.assertion import AssertionCreate, AssertionRead
from app.schemas.reextraction import (
    AssertionReExtractionApplyRead,
    AssertionReExtractionComparisonRead,
    ReExtractionApplyRequest,
    ReExtractionCompareRequest,
)
from app.services.parsing.normalization import normalize_for_match
from app.services.workflows.reextract_claims import (
    AssertionReExtractionApplyResult,
    AssertionReExtractionComparison,
    ClaimReExtractionService,
)

router = APIRouter(prefix="/api/v1", tags=["assertions"])


@router.post("/drafts/{draft_id}/assertions", response_model=AssertionRead, status_code=status.HTTP_201_CREATED)
def create_assertion(
    draft_id: UUID,
    payload: AssertionCreate,
    db: Session = Depends(get_db_session),
):
    if DraftRepository(db).get(draft_id) is None:
        raise NotFoundError("Draft not found.")

    assertion = AssertionRepository(db).create(
        draft_id,
        paragraph_index=payload.paragraph_index,
        sentence_index=payload.sentence_index,
        raw_text=payload.raw_text,
        normalized_text=normalize_for_match(payload.raw_text),
    )
    db.commit()
    db.refresh(assertion)
    return assertion


@router.get("/assertions/{assertion_id}", response_model=AssertionRead)
def get_assertion(assertion_id: UUID, db: Session = Depends(get_db_session)):
    assertion = AssertionRepository(db).get(assertion_id)
    if assertion is None:
        raise NotFoundError("Assertion not found.")
    return assertion


@router.post(
    "/assertions/{assertion_id}/reextract/compare",
    response_model=AssertionReExtractionComparisonRead,
)
def compare_reextraction(
    assertion_id: UUID,
    payload: ReExtractionCompareRequest | None = None,
    db: Session = Depends(get_db_session),
):
    if AssertionRepository(db).get(assertion_id) is None:
        raise NotFoundError("Assertion not found.")

    comparison = ClaimReExtractionService(db).compare_assertion(
        assertion_id,
        mode=(payload.mode if payload is not None else "structured"),
    )
    return _build_reextraction_comparison_response(comparison)


@router.post(
    "/assertions/{assertion_id}/reextract/apply",
    response_model=AssertionReExtractionApplyRead,
)
def apply_reextraction(
    assertion_id: UUID,
    payload: ReExtractionApplyRequest | None = None,
    db: Session = Depends(get_db_session),
):
    assertion_repo = AssertionRepository(db)
    if assertion_repo.get(assertion_id) is None:
        raise NotFoundError("Assertion not found.")

    result = ClaimReExtractionService(db).apply_assertion(
        assertion_id,
        mode=(payload.mode if payload is not None else "structured"),
    )
    db.commit()

    refreshed_assertion = assertion_repo.get(assertion_id)
    if refreshed_assertion is None:
        raise NotFoundError("Assertion not found.")
    return _build_reextraction_apply_response(result, refreshed_assertion)


def _build_reextraction_comparison_response(
    comparison: AssertionReExtractionComparison,
) -> dict[str, object]:
    return {
        "assertion_id": comparison.assertion_id,
        "existing_metadata": _serialize_extraction_metadata(comparison.existing_metadata),
        "proposed_metadata": {
            "strategy": comparison.proposed_metadata.strategy,
            "version": comparison.proposed_metadata.version,
        },
        "existing_claims": [_serialize_claim_preview(claim) for claim in comparison.existing_claims],
        "proposed_claims": [_serialize_claim_preview(claim) for claim in comparison.proposed_claims],
        "materially_changed": comparison.materially_changed,
        "apply_requires_replacement": comparison.apply_requires_replacement,
        "can_apply": comparison.can_apply,
        "blocked_reasons": list(comparison.blocked_reasons),
    }


def _build_reextraction_apply_response(
    result: AssertionReExtractionApplyResult,
    assertion: object,
) -> dict[str, object]:
    snapshot = assertion.extraction_snapshot if isinstance(assertion.extraction_snapshot, dict) else None
    snapshot_claims = snapshot.get("claims") if snapshot is not None else None
    notes = [
        "Extraction metadata and snapshot were refreshed.",
    ]
    if result.claims_replaced:
        notes.append("Persisted claim units were replaced during re-extraction.")
    else:
        notes.append("Existing persisted claim units already matched the selected extraction output.")

    return {
        "assertion_id": result.assertion_id,
        "applied_strategy": result.comparison.proposed_metadata.strategy,
        "applied_version": result.comparison.proposed_metadata.version,
        "materially_changed": result.comparison.materially_changed,
        "apply_requires_replacement": result.comparison.apply_requires_replacement,
        "claims_replaced": result.claims_replaced,
        "metadata_updated": result.metadata_updated,
        "resulting_claims": [_serialize_claim_preview(claim) for claim in result.claims],
        "updated_metadata": {
            "status": "versioned",
            "strategy": assertion.extraction_strategy,
            "version": assertion.extraction_version,
            "snapshot_present": assertion.extraction_snapshot is not None,
        },
        "snapshot_summary": {
            "present": assertion.extraction_snapshot is not None,
            "claim_count": len(snapshot_claims) if isinstance(snapshot_claims, list) else 0,
        },
        "notes": notes,
    }


def _serialize_extraction_metadata(metadata: object) -> dict[str, object]:
    return {
        "status": metadata.status,
        "strategy": metadata.strategy,
        "version": metadata.version,
        "snapshot_present": metadata.snapshot_present,
    }


def _serialize_claim_preview(claim: object) -> dict[str, object]:
    return {
        "claim_id": getattr(claim, "claim_id", None) or getattr(claim, "id", None),
        "text": claim.text,
        "normalized_text": claim.normalized_text,
        "claim_type": claim.claim_type,
        "sequence_order": claim.sequence_order,
    }
