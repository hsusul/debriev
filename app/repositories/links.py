"""Support link repository."""

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from app.models import SupportLink
from app.schemas.support_link import SupportLinkCreate
from app.services.linking.validation import LinkValidationService


class LinksRepository:
    """Persistence helpers for support links."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, claim_id: UUID, payload: SupportLinkCreate) -> SupportLink:
        LinkValidationService(self.session).validate_link_invariants(claim_id, payload.segment_id)

        existing = self.get_by_claim_and_segment(claim_id, payload.segment_id)
        if existing is not None:
            return existing

        link = SupportLink(
            claim_unit_id=claim_id,
            sequence_order=self._next_sequence_order(claim_id),
            **payload.model_dump(),
        )
        self.session.add(link)
        self.session.flush()
        return link

    def list_by_claim(self, claim_id: UUID) -> list[SupportLink]:
        stmt = (
            select(SupportLink)
            .options(joinedload(SupportLink.segment))
            .where(SupportLink.claim_unit_id == claim_id)
            .order_by(
                SupportLink.sequence_order.nulls_last(),
                SupportLink.created_at,
                SupportLink.id,
            )
        )
        return list(self.session.scalars(stmt))

    def get_by_claim_and_segment(self, claim_id: UUID, segment_id: UUID) -> SupportLink | None:
        stmt = (
            select(SupportLink)
            .options(joinedload(SupportLink.segment))
            .where(
                SupportLink.claim_unit_id == claim_id,
                SupportLink.segment_id == segment_id,
            )
        )
        return self.session.scalar(stmt)

    def _next_sequence_order(self, claim_id: UUID) -> int:
        stmt = select(func.max(SupportLink.sequence_order)).where(SupportLink.claim_unit_id == claim_id)
        max_sequence = self.session.scalar(stmt)
        return (max_sequence or 0) + 1
