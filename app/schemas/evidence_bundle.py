"""Evidence bundle schemas."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel


class EvidenceBundleCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    source_document_ids: list[UUID] = Field(default_factory=list)


class EvidenceBundleRead(ORMModel):
    id: UUID
    matter_id: UUID
    name: str
    description: str | None
    source_document_ids: list[UUID] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
