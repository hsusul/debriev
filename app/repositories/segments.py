"""Segment repository."""

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Assertion, ClaimUnit, Draft, Segment
from app.repositories.evidence_bundles import EvidenceBundleRepository


class SegmentRepository:
    """Persistence helpers for anchored segments."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create_many(
        self,
        source_document_id: UUID,
        segments: Sequence[Mapping[str, Any]],
    ) -> list[Segment]:
        records = [Segment(source_document_id=source_document_id, **dict(item)) for item in segments]
        self.session.add_all(records)
        self.session.flush()
        return records

    def list_by_source_document(self, source_document_id: UUID) -> list[Segment]:
        stmt = self._ordered_by_source_document_stmt(source_document_id)
        segments = list(self.session.scalars(stmt))
        return sorted(segments, key=lambda segment: segment.normalized_anchor_sort_key)

    def list_by_source_documents(self, source_document_ids: Sequence[UUID]) -> list[Segment]:
        unique_ids = list(dict.fromkeys(source_document_ids))
        if not unique_ids:
            return []

        stmt = select(Segment).where(Segment.source_document_id.in_(unique_ids))
        source_order = {source_document_id: index for index, source_document_id in enumerate(unique_ids)}
        segments = list(self.session.scalars(stmt))
        return sorted(
            segments,
            key=lambda segment: (
                source_order.get(segment.source_document_id, len(unique_ids)),
                segment.normalized_anchor_sort_key,
            ),
        )

    def get(self, segment_id: UUID) -> Segment | None:
        return self.session.get(Segment, segment_id)

    def list_candidate_segments_for_draft(
        self,
        draft_id: UUID,
        *,
        fallback_to_matter_sources: bool = True,
    ) -> list[Segment]:
        allowed_source_document_ids = EvidenceBundleRepository(self.session).resolve_allowed_source_document_ids_for_draft(
            draft_id,
            fallback_to_matter_sources=fallback_to_matter_sources,
        )
        return self.list_by_source_documents(allowed_source_document_ids)

    def list_candidate_segments_for_claim(
        self,
        claim_id: UUID,
        *,
        fallback_to_matter_sources: bool = True,
    ) -> list[Segment]:
        draft_id = self._resolve_draft_id_for_claim(claim_id)
        if draft_id is None:
            return []
        return self.list_candidate_segments_for_draft(
            draft_id,
            fallback_to_matter_sources=fallback_to_matter_sources,
        )

    def get_local_context_window(self, segment_id: UUID, *, radius: int = 1) -> list[Segment]:
        target_segment = self.get(segment_id)
        if target_segment is None:
            return []

        ordered_segments = self.list_by_source_document(target_segment.source_document_id)
        try:
            center_index = next(
                index
                for index, segment in enumerate(ordered_segments)
                if segment.id == target_segment.id
            )
        except StopIteration:
            return []

        start_index = max(0, center_index - radius)
        end_index = center_index + radius + 1
        return ordered_segments[start_index:end_index]

    def _ordered_by_source_document_stmt(self, source_document_id: UUID):
        return select(Segment).where(Segment.source_document_id == source_document_id)

    def _resolve_draft_id_for_claim(self, claim_id: UUID) -> UUID | None:
        stmt = (
            select(Draft.id)
            .join(Assertion, Assertion.draft_id == Draft.id)
            .join(ClaimUnit, ClaimUnit.assertion_id == Assertion.id)
            .where(ClaimUnit.id == claim_id)
        )
        return self.session.scalar(stmt)
