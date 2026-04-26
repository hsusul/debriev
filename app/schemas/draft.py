"""Draft schemas."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.core.enums import DraftMode
from app.schemas.common import ORMModel


class DraftCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    mode: DraftMode = DraftMode.DRAFT
    evidence_bundle_id: UUID | None = None


class DraftRead(ORMModel):
    id: UUID
    matter_id: UUID
    evidence_bundle_id: UUID | None
    title: str
    mode: DraftMode
    created_at: datetime
    updated_at: datetime
