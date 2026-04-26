"""Source document repository."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import ParserStatus
from app.models import SourceDocument
from app.schemas.source_document import SourceDocumentCreate


class SourceRepository:
    """Persistence helpers for source documents."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, matter_id: UUID, payload: SourceDocumentCreate) -> SourceDocument:
        data = payload.model_dump(exclude={"content"})
        source = SourceDocument(
            matter_id=matter_id,
            parser_status=ParserStatus.PROCESSING if payload.content else ParserStatus.PENDING,
            **data,
        )
        self.session.add(source)
        self.session.flush()
        return source

    def get(self, source_id: UUID) -> SourceDocument | None:
        return self.session.get(SourceDocument, source_id)

    def list_by_matter(self, matter_id: UUID) -> list[SourceDocument]:
        return list(
            self.session.scalars(
                select(SourceDocument).where(SourceDocument.matter_id == matter_id)
            )
        )
