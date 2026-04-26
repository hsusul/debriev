"""Source document and segment routes."""

from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.core.exceptions import NotFoundError
from app.repositories.segments import SegmentRepository
from app.repositories.sources import SourceRepository
from app.schemas.segment import SegmentRead
from app.schemas.source_document import SourceDocumentCreate, SourceDocumentRead
from app.services.parsing.ingestion import EvidenceIngestionService

router = APIRouter(prefix="/api/v1", tags=["sources"])


@router.post("/matters/{matter_id}/sources", response_model=SourceDocumentRead, status_code=status.HTTP_201_CREATED)
def create_source(
    matter_id: UUID,
    payload: SourceDocumentCreate,
    db: Session = Depends(get_db_session),
):
    service = EvidenceIngestionService(db)
    source = service.create_source_with_segments(matter_id, payload)
    db.commit()
    db.refresh(source)
    return source


@router.get("/sources/{source_id}", response_model=SourceDocumentRead)
def get_source(source_id: UUID, db: Session = Depends(get_db_session)):
    source = SourceRepository(db).get(source_id)
    if source is None:
        raise NotFoundError("Source document not found.")
    return source


@router.get("/sources/{source_id}/segments", response_model=list[SegmentRead])
def list_segments(source_id: UUID, db: Session = Depends(get_db_session)):
    source = SourceRepository(db).get(source_id)
    if source is None:
        raise NotFoundError("Source document not found.")
    return SegmentRepository(db).list_by_source_document(source_id)
