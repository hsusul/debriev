"""Source-agnostic ingestion orchestration."""

from uuid import UUID

from sqlalchemy.orm import Session

from app.core.enums import ParserStatus
from app.core.exceptions import NotFoundError
from app.repositories.matters import MatterRepository
from app.repositories.segments import SegmentRepository
from app.repositories.sources import SourceRepository
from app.schemas.source_document import SourceDocumentCreate
from app.services.parsing.registry import ParserRegistry, build_default_parser_registry


class EvidenceIngestionService:
    """Coordinate source creation, parser selection, parsing, and segment persistence."""

    def __init__(
        self,
        session: Session,
        *,
        parser_registry: ParserRegistry | None = None,
    ) -> None:
        self.session = session
        self.parser_registry = parser_registry or build_default_parser_registry()
        self.matters = MatterRepository(session)
        self.sources = SourceRepository(session)
        self.segments = SegmentRepository(session)

    def create_source_with_segments(self, matter_id: UUID, payload: SourceDocumentCreate):
        matter = self.matters.get(matter_id)
        if matter is None:
            raise NotFoundError("Matter not found.")

        parser = None
        if payload.content:
            parser = self.parser_registry.get(payload.source_type)

        source = self.sources.create(matter_id, payload)
        if not payload.content:
            return source

        # TODO: replace this plain-text path with file-backed OCR/PDF extraction once source ingestion expands.
        parsed_segments = parser.parse(payload.content)
        if parsed_segments:
            self.segments.create_many(source.id, [segment.to_record() for segment in parsed_segments])
            source.parser_status = ParserStatus.COMPLETED
            source.parser_confidence = parser.confidence_for_segments(parsed_segments)
        else:
            source.parser_status = ParserStatus.FAILED
            source.parser_confidence = 0.0

        self.session.flush()
        return source


class TranscriptIngestionService(EvidenceIngestionService):
    """Temporary compatibility shim while imports move to EvidenceIngestionService."""
