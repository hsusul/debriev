"""Audit routes."""

from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.core.exceptions import NotFoundError
from app.repositories.decisions import DecisionsRepository
from app.repositories.verification import VerificationRepository
from app.schemas.audit import AuditReportRead
from app.services.audit.report_builder import build_audit_summary

router = APIRouter(prefix="/api/v1", tags=["audit"])


@router.get("/verification-runs/{run_id}/audit", response_model=AuditReportRead)
def get_audit_report(run_id: UUID, db: Session = Depends(get_db_session)):
    run = VerificationRepository(db).get(run_id)
    if run is None:
        raise NotFoundError("Verification run not found.")

    decisions = DecisionsRepository(db).list_by_run(run_id)
    return {
        "run": run,
        "decisions": decisions,
        "summary": build_audit_summary(run, decisions),
    }

