"""Evidence bundle repository."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.exceptions import NotFoundError, ValidationError
from app.models import Draft, EvidenceBundle, SourceDocument
from app.models.evidence_bundle import evidence_bundle_source_documents
from app.schemas.evidence_bundle import EvidenceBundleCreate


class EvidenceBundleRepository:
    """Persistence helpers for explicit draft evidence scopes."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, matter_id: UUID, payload: EvidenceBundleCreate) -> EvidenceBundle:
        bundle = EvidenceBundle(
            matter_id=matter_id,
            name=payload.name,
            description=payload.description,
        )
        self.session.add(bundle)
        self.session.flush()

        if payload.source_document_ids:
            bundle.source_documents.extend(self._load_source_documents(matter_id, payload.source_document_ids))
            self.session.flush()

        return bundle

    def get(self, bundle_id: UUID) -> EvidenceBundle | None:
        stmt = (
            select(EvidenceBundle)
            .options(selectinload(EvidenceBundle.source_documents))
            .where(EvidenceBundle.id == bundle_id)
        )
        return self.session.scalar(stmt)

    def add_source_document(self, bundle_id: UUID, source_document_id: UUID) -> EvidenceBundle:
        bundle = self.get(bundle_id)
        if bundle is None:
            raise NotFoundError("Evidence bundle not found.")

        source_document = self.session.get(SourceDocument, source_document_id)
        if source_document is None:
            raise NotFoundError("Source document not found.")
        if source_document.matter_id != bundle.matter_id:
            raise ValidationError("Source document must belong to the same matter as the evidence bundle.")

        if all(source.id != source_document.id for source in bundle.source_documents):
            bundle.source_documents.append(source_document)
            self.session.flush()
        return bundle

    def list_source_documents(self, bundle_id: UUID) -> list[SourceDocument]:
        bundle = self.get(bundle_id)
        if bundle is None:
            return []
        return self._ordered_source_documents(bundle.source_documents)

    def contains_source_document(self, bundle_id: UUID, source_document_id: UUID) -> bool:
        stmt = select(evidence_bundle_source_documents.c.source_document_id).where(
            evidence_bundle_source_documents.c.evidence_bundle_id == bundle_id,
            evidence_bundle_source_documents.c.source_document_id == source_document_id,
        )
        return self.session.scalar(stmt) is not None

    def resolve_allowed_source_document_ids_for_draft(
        self,
        draft_id: UUID,
        *,
        fallback_to_matter_sources: bool = True,
    ) -> list[UUID]:
        draft = self._get_draft_with_scope(draft_id)
        if draft is None:
            return []

        if draft.evidence_bundle is not None:
            source_documents = self._ordered_source_documents(draft.evidence_bundle.source_documents)
            return [source.id for source in source_documents]

        if not fallback_to_matter_sources:
            return []

        source_documents = self.session.scalars(
            select(SourceDocument).where(SourceDocument.matter_id == draft.matter_id)
        ).all()
        return [source.id for source in self._ordered_source_documents(source_documents)]

    def resolve_source_document_ids_for_draft(
        self,
        draft_id: UUID,
        *,
        fallback_to_matter_sources: bool = True,
    ) -> list[UUID]:
        """Compatibility shim for older call sites."""

        return self.resolve_allowed_source_document_ids_for_draft(
            draft_id,
            fallback_to_matter_sources=fallback_to_matter_sources,
        )

    def _get_draft_with_scope(self, draft_id: UUID) -> Draft | None:
        stmt = (
            select(Draft)
            .options(selectinload(Draft.evidence_bundle).selectinload(EvidenceBundle.source_documents))
            .where(Draft.id == draft_id)
        )
        return self.session.scalar(stmt)

    def _load_source_documents(self, matter_id: UUID, source_document_ids: list[UUID]) -> list[SourceDocument]:
        unique_ids = list(dict.fromkeys(source_document_ids))
        source_documents = self.session.scalars(
            select(SourceDocument).where(SourceDocument.id.in_(unique_ids))
        ).all()
        if len(source_documents) != len(unique_ids):
            raise ValidationError("All source documents in an evidence bundle must exist.")
        if any(source_document.matter_id != matter_id for source_document in source_documents):
            raise ValidationError("All source documents in an evidence bundle must belong to the same matter.")
        return self._ordered_source_documents(source_documents)

    def _ordered_source_documents(self, source_documents: list[SourceDocument]) -> list[SourceDocument]:
        return sorted(
            source_documents,
            key=lambda source_document: (
                source_document.created_at or datetime.min,
                source_document.file_name,
                str(source_document.id),
            ),
        )
