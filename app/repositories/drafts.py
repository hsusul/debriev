"""Draft repository."""

from uuid import UUID

from sqlalchemy.orm import Session

from app.core.exceptions import NotFoundError, ValidationError
from app.models import Draft
from app.schemas.draft import DraftCreate
from app.repositories.evidence_bundles import EvidenceBundleRepository


class DraftRepository:
    """Persistence helpers for drafts."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, matter_id: UUID, payload: DraftCreate) -> Draft:
        if payload.evidence_bundle_id is not None:
            bundle = EvidenceBundleRepository(self.session).get(payload.evidence_bundle_id)
            if bundle is None:
                raise NotFoundError("Evidence bundle not found.")
            if bundle.matter_id != matter_id:
                raise ValidationError("Draft evidence bundle must belong to the same matter.")

        draft = Draft(matter_id=matter_id, **payload.model_dump())
        self.session.add(draft)
        self.session.flush()
        return draft

    def get(self, draft_id: UUID) -> Draft | None:
        return self.session.get(Draft, draft_id)

    def set_evidence_bundle(self, draft_id: UUID, evidence_bundle_id: UUID | None) -> Draft:
        draft = self.get(draft_id)
        if draft is None:
            raise NotFoundError("Draft not found.")

        if evidence_bundle_id is None:
            draft.evidence_bundle_id = None
            self.session.flush()
            return draft

        bundle = EvidenceBundleRepository(self.session).get(evidence_bundle_id)
        if bundle is None:
            raise NotFoundError("Evidence bundle not found.")
        if bundle.matter_id != draft.matter_id:
            raise ValidationError("Draft evidence bundle must belong to the same matter.")

        draft.evidence_bundle_id = evidence_bundle_id
        self.session.flush()
        return draft
