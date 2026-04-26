"""Claim repository."""

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.enums import SupportStatus
from app.models import Assertion, ClaimUnit


class ClaimsRepository:
    """Persistence helpers for claim units."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create_many(self, assertion_id: UUID, claims: Sequence[object]) -> list[ClaimUnit]:
        records = [
            ClaimUnit(
                assertion_id=assertion_id,
                text=claim.text,
                normalized_text=claim.normalized_text,
                claim_type=claim.claim_type,
                sequence_order=claim.sequence_order,
                support_status=SupportStatus.UNVERIFIED,
            )
            for claim in claims
        ]
        self.session.add_all(records)
        self.session.flush()
        return records

    def get(self, claim_id: UUID) -> ClaimUnit | None:
        return self.session.get(ClaimUnit, claim_id)

    def list_by_assertion(self, assertion_id: UUID) -> list[ClaimUnit]:
        stmt = select(ClaimUnit).where(ClaimUnit.assertion_id == assertion_id).order_by(ClaimUnit.sequence_order)
        return list(self.session.scalars(stmt))

    def list_by_draft(self, draft_id: UUID) -> list[ClaimUnit]:
        stmt = (
            select(ClaimUnit)
            .join(ClaimUnit.assertion)
            .options(
                selectinload(ClaimUnit.assertion),
                selectinload(ClaimUnit.verification_runs),
                selectinload(ClaimUnit.review_decisions),
            )
            .where(Assertion.draft_id == draft_id)
        )
        claims = list(self.session.scalars(stmt))
        return sorted(
            claims,
            key=lambda claim: (
                claim.assertion.paragraph_index is None,
                claim.assertion.paragraph_index or 0,
                claim.assertion.sentence_index is None,
                claim.assertion.sentence_index or 0,
                claim.sequence_order,
                claim.created_at,
                str(claim.id),
            ),
        )

    def replace_for_assertion(self, assertion_id: UUID, claims: Sequence[object]) -> list[ClaimUnit]:
        existing = self.list_by_assertion(assertion_id)
        for claim in existing:
            self.session.delete(claim)
        self.session.flush()
        return self.create_many(assertion_id, claims)
