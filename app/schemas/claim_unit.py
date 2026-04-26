"""Claim unit schemas."""

from datetime import datetime
from uuid import UUID

from app.core.enums import ClaimType, SupportStatus
from app.schemas.common import ORMModel


class ClaimUnitRead(ORMModel):
    id: UUID
    assertion_id: UUID
    text: str
    normalized_text: str
    claim_type: ClaimType
    sequence_order: int
    support_status: SupportStatus
    created_at: datetime
    updated_at: datetime

