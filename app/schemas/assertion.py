"""Assertion schemas."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel


class AssertionCreate(BaseModel):
    paragraph_index: int | None = None
    sentence_index: int | None = None
    raw_text: str = Field(..., min_length=1)


class AssertionRead(ORMModel):
    id: UUID
    draft_id: UUID
    paragraph_index: int | None
    sentence_index: int | None
    raw_text: str
    normalized_text: str
    created_at: datetime
    updated_at: datetime

