"""Assertion repository."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Assertion


class AssertionRepository:
    """Persistence helpers for assertions."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        draft_id: UUID,
        *,
        paragraph_index: int | None,
        sentence_index: int | None,
        raw_text: str,
        normalized_text: str,
    ) -> Assertion:
        assertion = Assertion(
            draft_id=draft_id,
            paragraph_index=paragraph_index,
            sentence_index=sentence_index,
            raw_text=raw_text,
            normalized_text=normalized_text,
        )
        self.session.add(assertion)
        self.session.flush()
        return assertion

    def get(self, assertion_id: UUID) -> Assertion | None:
        return self.session.get(Assertion, assertion_id)

    def list_by_draft(self, draft_id: UUID) -> list[Assertion]:
        assertions = list(
            self.session.scalars(
                select(Assertion).where(Assertion.draft_id == draft_id)
            )
        )
        return sorted(
            assertions,
            key=lambda assertion: (
                assertion.paragraph_index is None,
                assertion.paragraph_index or 0,
                assertion.sentence_index is None,
                assertion.sentence_index or 0,
                assertion.created_at,
                str(assertion.id),
            ),
        )

    def record_extraction(
        self,
        assertion: Assertion,
        *,
        strategy: str,
        version: int,
        snapshot: dict[str, object],
    ) -> Assertion:
        assertion.extraction_strategy = strategy
        assertion.extraction_version = version
        assertion.extraction_snapshot = snapshot
        self.session.add(assertion)
        self.session.flush()
        return assertion
