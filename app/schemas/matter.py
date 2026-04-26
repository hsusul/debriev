"""Matter schemas."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel


class MatterCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    court: str | None = Field(default=None, max_length=255)
    jurisdiction: str | None = Field(default=None, max_length=255)
    status: str = Field(default="ACTIVE", max_length=50)


class MatterRead(ORMModel):
    id: UUID
    name: str
    court: str | None
    jurisdiction: str | None
    status: str
    created_at: datetime
    updated_at: datetime

