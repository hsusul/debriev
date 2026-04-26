"""Claim graph edge repository."""

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import ClaimGraphRelationshipType
from app.models import ClaimGraphEdge


@dataclass(slots=True)
class ClaimGraphEdgeCreate:
    draft_id: UUID
    draft_review_run_id: UUID
    source_claim_id: UUID
    target_claim_id: UUID
    relationship_type: ClaimGraphRelationshipType
    reason_code: str | None = None
    reason_text: str | None = None
    confidence_score: float | None = None


class ClaimGraphEdgeRepository:
    """Persistence helpers for immutable draft-review claim graph edges."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create_many(self, items: Sequence[ClaimGraphEdgeCreate]) -> list[ClaimGraphEdge]:
        records = [
            ClaimGraphEdge(
                draft_id=item.draft_id,
                draft_review_run_id=item.draft_review_run_id,
                source_claim_id=item.source_claim_id,
                target_claim_id=item.target_claim_id,
                relationship_type=item.relationship_type,
                reason_code=item.reason_code,
                reason_text=item.reason_text,
                confidence_score=item.confidence_score,
            )
            for item in items
        ]
        if records:
            self.session.add_all(records)
            self.session.flush()
        return records

    def list_by_review_run(self, draft_review_run_id: UUID) -> list[ClaimGraphEdge]:
        stmt = (
            select(ClaimGraphEdge)
            .where(ClaimGraphEdge.draft_review_run_id == draft_review_run_id)
            .order_by(
                ClaimGraphEdge.relationship_type.asc(),
                ClaimGraphEdge.source_claim_id.asc(),
                ClaimGraphEdge.target_claim_id.asc(),
            )
        )
        return list(self.session.scalars(stmt))
