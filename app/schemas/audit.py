"""Audit report schemas."""

from app.schemas.common import ORMModel
from app.schemas.verification import UserDecisionRead, VerificationRunRead


class AuditReportRead(ORMModel):
    run: VerificationRunRead
    decisions: list[UserDecisionRead]
    summary: str

