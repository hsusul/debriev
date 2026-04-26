"""Segment schemas."""

from datetime import datetime
from uuid import UUID

from app.schemas.common import ORMModel


class SegmentRead(ORMModel):
    id: UUID
    source_document_id: UUID
    page_start: int | None
    line_start: int | None
    page_end: int | None
    line_end: int | None
    raw_text: str
    normalized_text: str
    speaker: str | None
    segment_type: str
    created_at: datetime

