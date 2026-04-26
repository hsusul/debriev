"""Source document schemas."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.core.enums import ParserStatus, SourceType
from app.schemas.common import ORMModel


class SourceDocumentCreate(BaseModel):
    file_name: str = Field(..., min_length=1, max_length=255)
    source_type: SourceType = SourceType.DEPOSITION
    raw_file_path: str = Field(..., min_length=1)
    content: str | None = Field(
        default=None,
        description="Optional raw transcript text to parse into anchored segments.",
    )


class SourceDocumentRead(ORMModel):
    id: UUID
    matter_id: UUID
    file_name: str
    source_type: SourceType
    raw_file_path: str
    parser_status: ParserStatus
    parser_confidence: float | None
    created_at: datetime
    updated_at: datetime

