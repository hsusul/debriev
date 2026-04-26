"""Support link schemas."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.core.enums import LinkType
from app.schemas.common import ORMModel


class SupportLinkCreate(BaseModel):
    segment_id: UUID
    link_type: LinkType = LinkType.MANUAL
    citation_text: str | None = None
    user_confirmed: bool = True


class SupportLinkRead(ORMModel):
    id: UUID
    claim_unit_id: UUID
    segment_id: UUID
    link_type: LinkType
    citation_text: str | None
    user_confirmed: bool
    created_at: datetime

