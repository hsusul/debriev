import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.exceptions import ValidationError
from app.core.provenance import BLOCK_KIND, EXHIBIT_PAGE_KIND, PAGE_LINE_KIND, PARAGRAPH_KIND
from app.core.enums import ParserStatus, SourceType
from app.models import Base, Matter
from app.repositories.segments import SegmentRepository
from app.schemas.source_document import SourceDocumentCreate
from app.services.parsing.base import ParsedSegment
from app.services.parsing.ingestion import EvidenceIngestionService
from app.services.parsing.registry import ParserRegistry, build_default_parser_registry
from app.services.parsing.transcript_parser import DeclarationParser, ExhibitParser, TranscriptParser


@pytest.fixture
def session() -> Session:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    db_session = SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()
        engine.dispose()


def test_transcript_parser_extracts_page_line_anchors() -> None:
    sample = """
    Page 12
    1 Q. Please state your name.
    2 A. John Doe.
    3 A. I signed the contract.
    Page 13
    1 A. The email was sent the same day.
    """

    segments = TranscriptParser().parse(sample)

    assert len(segments) == 3
    assert segments[0].page_start == 12
    assert segments[0].line_start == 1
    assert segments[0].normalized_anchor_metadata == {
        "kind": PAGE_LINE_KIND,
        "page_start": 12,
        "line_start": 1,
        "page_end": 12,
        "line_end": 1,
    }
    assert segments[0].rendered_anchor == "p.12:1-12:1"
    assert segments[0].speaker == "Q"
    assert segments[0].segment_type == "QUESTION_BLOCK"
    assert segments[1].line_start == 2
    assert segments[1].line_end == 3
    assert segments[1].rendered_anchor == "p.12:2-12:3"
    assert segments[1].speaker == "A"
    assert segments[1].segment_type == "ANSWER_BLOCK"
    assert segments[2].page_start == 13
    assert "email was sent" in segments[2].normalized_text


def test_transcript_parser_keeps_local_unanchored_runs_separate() -> None:
    sample = """
    Preliminary note.
    Another preliminary note.
    Page 8
    1 Q. State your name.
    2 A. John Doe.
    Off the record.
    Break taken.
    3 A. Back on the record.
    """

    segments = TranscriptParser().parse(sample)

    assert len(segments) == 5
    assert segments[0].segment_type == "UNANCHORED_TEXT"
    assert segments[0].raw_text == "Preliminary note.\nAnother preliminary note."
    assert segments[0].normalized_anchor_metadata is None
    assert segments[0].rendered_anchor == "unanchored segment"
    assert segments[1].segment_type == "QUESTION_BLOCK"
    assert segments[2].segment_type == "ANSWER_BLOCK"
    assert segments[3].segment_type == "UNANCHORED_TEXT"
    assert segments[3].raw_text == "Off the record.\nBreak taken."
    assert segments[4].segment_type == "ANSWER_BLOCK"
    assert segments[4].page_start == 8
    assert segments[4].line_start == 3


def test_declaration_parser_extracts_numbered_paragraph_segments() -> None:
    sample = """
    1. I am over eighteen years old.
    2. On March 1, 2024, I signed the agreement
       and returned it to counsel.
    3. I declare under penalty of perjury that the foregoing is true.
    """

    segments = DeclarationParser().parse(sample)

    assert len(segments) == 3
    assert segments[0].page_start is None
    assert segments[0].line_start is None
    assert segments[0].speaker == "DECLARANT ¶1"
    assert segments[0].segment_type == "DECLARATION_PARAGRAPH"
    assert segments[0].normalized_anchor_metadata == {
        "kind": PARAGRAPH_KIND,
        "paragraph_number": 1,
        "sequence_order": 1,
    }
    assert segments[0].rendered_anchor == "¶1"
    assert segments[1].normalized_anchor_metadata == {
        "kind": PARAGRAPH_KIND,
        "paragraph_number": 2,
        "sequence_order": 2,
    }
    assert segments[1].rendered_anchor == "¶2"
    assert segments[1].raw_text == "On March 1, 2024, I signed the agreement\nand returned it to counsel."
    assert segments[1].normalized_text == "on march 1, 2024, i signed the agreement and returned it to counsel."
    assert segments[2].speaker == "DECLARANT ¶3"
    assert segments[2].rendered_anchor == "¶3"


def test_declaration_parser_uses_stable_blocks_without_numbering() -> None:
    sample = """
    I, Jane Doe, declare as follows.

    I reviewed the agreement on March 1.
    I signed the declaration on March 2.

    I declare under penalty of perjury that the foregoing is true.
    """

    segments = DeclarationParser().parse(sample)

    assert len(segments) == 3
    assert all(segment.line_start is None for segment in segments)
    assert all(segment.page_start is None for segment in segments)
    assert [segment.segment_type for segment in segments] == [
        "DECLARATION_BLOCK",
        "DECLARATION_BLOCK",
        "DECLARATION_BLOCK",
    ]
    assert [segment.speaker for segment in segments] == [
        "DECLARANT",
        "DECLARANT",
        "DECLARANT",
    ]
    assert [segment.normalized_anchor_metadata for segment in segments] == [
        {"kind": BLOCK_KIND, "block_index": 1, "sequence_order": 1},
        {"kind": BLOCK_KIND, "block_index": 2, "sequence_order": 2},
        {"kind": BLOCK_KIND, "block_index": 3, "sequence_order": 3},
    ]
    assert [segment.rendered_anchor for segment in segments] == ["block 1", "block 2", "block 3"]


def test_ingestion_service_parses_declaration_sources_into_segments(session: Session) -> None:
    matter = Matter(name="Acme v. Doe", status="ACTIVE")
    session.add(matter)
    session.commit()

    payload = SourceDocumentCreate(
        file_name="declaration.txt",
        source_type=SourceType.DECLARATION,
        raw_file_path="/tmp/declaration.txt",
        content="""
        1. I reviewed the agreement.
        2. I signed the declaration.
        """,
    )

    source = EvidenceIngestionService(session).create_source_with_segments(matter.id, payload)
    session.commit()

    segments = SegmentRepository(session).list_by_source_document(source.id)

    assert source.source_type == SourceType.DECLARATION
    assert source.parser_status == ParserStatus.COMPLETED
    assert source.parser_confidence == 0.85
    assert [segment.raw_text for segment in segments] == [
        "I reviewed the agreement.",
        "I signed the declaration.",
    ]
    assert [segment.speaker for segment in segments] == [
        "DECLARANT ¶1",
        "DECLARANT ¶2",
    ]
    assert [segment.segment_type for segment in segments] == [
        "DECLARATION_PARAGRAPH",
        "DECLARATION_PARAGRAPH",
    ]
    assert [segment.page_start for segment in segments] == [None, None]
    assert [segment.line_start for segment in segments] == [None, None]
    assert [segment.anchor_metadata for segment in segments] == [
        {"kind": PARAGRAPH_KIND, "paragraph_number": 1, "sequence_order": 1},
        {"kind": PARAGRAPH_KIND, "paragraph_number": 2, "sequence_order": 2},
    ]
    assert [segment.rendered_anchor for segment in segments] == ["¶1", "¶2"]


def test_ingestion_service_parses_deposition_sources_into_segments(session: Session) -> None:
    matter = Matter(name="Acme v. Doe", status="ACTIVE")
    session.add(matter)
    session.commit()

    payload = SourceDocumentCreate(
        file_name="deposition.txt",
        source_type=SourceType.DEPOSITION,
        raw_file_path="/tmp/deposition.txt",
        content="""
        Page 10
        1 Q. Did you sign the agreement?
        2 A. Yes.
        3 A. I signed it on March 1.
        """,
    )

    source = EvidenceIngestionService(session).create_source_with_segments(matter.id, payload)
    session.commit()

    segments = SegmentRepository(session).list_by_source_document(source.id)

    assert source.source_type == SourceType.DEPOSITION
    assert source.parser_status == ParserStatus.COMPLETED
    assert source.parser_confidence == 0.95
    assert [segment.segment_type for segment in segments] == ["QUESTION_BLOCK", "ANSWER_BLOCK"]
    assert [segment.rendered_anchor for segment in segments] == ["p.10:1-10:1", "p.10:2-10:3"]
    assert segments[1].raw_text == "A. Yes.\nA. I signed it on March 1."


def test_exhibit_parser_extracts_page_and_labeled_blocks() -> None:
    sample = """
    Exhibit Page 1
    Subject: Contract Status
    The contract was signed on March 1.

    Notes
    Payment was received on March 3.

    Exhibit Page 2
    Date: March 4, 2024
    A follow-up notice was sent to counsel.
    """

    segments = ExhibitParser().parse(sample)

    assert len(segments) == 5
    assert [segment.segment_type for segment in segments] == [
        "EXHIBIT_LABELED_BLOCK",
        "EXHIBIT_PAGE_BLOCK",
        "EXHIBIT_PAGE_BLOCK",
        "EXHIBIT_LABELED_BLOCK",
        "EXHIBIT_PAGE_BLOCK",
    ]
    assert segments[0].speaker == "Subject"
    assert segments[0].normalized_anchor_metadata == {
        "kind": EXHIBIT_PAGE_KIND,
        "page_number": 1,
        "block_index": 1,
        "sequence_order": 1,
        "label": "Subject",
    }
    assert segments[0].rendered_anchor == "ex. p.1 block 1 [Subject]"
    assert segments[1].rendered_anchor == "ex. p.1 block 2"
    assert segments[1].raw_text == "The contract was signed on March 1."
    assert segments[2].rendered_anchor == "ex. p.1 block 3"
    assert segments[2].raw_text == "Notes\nPayment was received on March 3."
    assert segments[3].rendered_anchor == "ex. p.2 block 1 [Date]"
    assert segments[4].normalized_text == "a follow-up notice was sent to counsel."


def test_exhibit_parser_uses_stable_fallback_blocks_without_pages() -> None:
    sample = """
    Exhibit A

    The attached invoice reflects a balance of $500.

    From: Accounts Receivable
    """

    segments = ExhibitParser().parse(sample)

    assert len(segments) == 3
    assert [segment.segment_type for segment in segments] == [
        "EXHIBIT_TEXT_BLOCK",
        "EXHIBIT_TEXT_BLOCK",
        "EXHIBIT_LABELED_BLOCK",
    ]
    assert [segment.normalized_anchor_metadata for segment in segments] == [
        {"kind": BLOCK_KIND, "block_index": 1, "sequence_order": 1},
        {"kind": BLOCK_KIND, "block_index": 2, "sequence_order": 2},
        {"kind": BLOCK_KIND, "block_index": 3, "sequence_order": 3, "label": "From"},
    ]
    assert [segment.rendered_anchor for segment in segments] == [
        "block 1",
        "block 2",
        "block 3 [From]",
    ]


def test_ingestion_service_parses_exhibit_sources_into_segments(session: Session) -> None:
    matter = Matter(name="Acme v. Doe", status="ACTIVE")
    session.add(matter)
    session.commit()

    payload = SourceDocumentCreate(
        file_name="exhibit.txt",
        source_type=SourceType.EXHIBIT,
        raw_file_path="/tmp/exhibit.txt",
        content="""
        Page 1
        Subject: Contract Status
        The contract was signed on March 1.

        Page 2
        A follow-up notice was sent to counsel.
        """,
    )

    source = EvidenceIngestionService(session).create_source_with_segments(matter.id, payload)
    session.commit()

    segments = SegmentRepository(session).list_by_source_document(source.id)

    assert source.source_type == SourceType.EXHIBIT
    assert source.parser_status == ParserStatus.COMPLETED
    assert source.parser_confidence == 0.8
    assert [segment.segment_type for segment in segments] == [
        "EXHIBIT_LABELED_BLOCK",
        "EXHIBIT_PAGE_BLOCK",
        "EXHIBIT_PAGE_BLOCK",
    ]
    assert [segment.rendered_anchor for segment in segments] == [
        "ex. p.1 block 1 [Subject]",
        "ex. p.1 block 2",
        "ex. p.2 block 1",
    ]
    assert [segment.page_start for segment in segments] == [None, None, None]


def test_parser_registry_returns_registered_parsers_by_source_type() -> None:
    registry = build_default_parser_registry()

    deposition_parser = registry.get(SourceType.DEPOSITION)
    declaration_parser = registry.get(SourceType.DECLARATION)
    exhibit_parser = registry.get(SourceType.EXHIBIT)

    assert isinstance(deposition_parser, TranscriptParser)
    assert isinstance(declaration_parser, DeclarationParser)
    assert isinstance(exhibit_parser, ExhibitParser)
    assert deposition_parser.source_type == SourceType.DEPOSITION
    assert declaration_parser.source_type == SourceType.DECLARATION
    assert exhibit_parser.source_type == SourceType.EXHIBIT


def test_ingestion_service_uses_parser_registry(session: Session) -> None:
    matter = Matter(name="Acme v. Doe", status="ACTIVE")
    session.add(matter)
    session.commit()

    parser = SpyParser(
        source_type=SourceType.DECLARATION,
        segments=[
            ParsedSegment(
                page_start=None,
                line_start=None,
                page_end=None,
                line_end=None,
                anchor_metadata={"kind": PARAGRAPH_KIND, "paragraph_number": 7, "sequence_order": 1},
                raw_text="I signed the declaration.",
                normalized_text="i signed the declaration.",
                speaker="DECLARANT ¶7",
                segment_type="DECLARATION_PARAGRAPH",
            )
        ],
        confidence=0.77,
    )
    registry = SpyRegistry(parser)
    payload = SourceDocumentCreate(
        file_name="declaration.txt",
        source_type=SourceType.DECLARATION,
        raw_file_path="/tmp/declaration.txt",
        content="7. I signed the declaration.",
    )

    source = EvidenceIngestionService(session, parser_registry=registry).create_source_with_segments(matter.id, payload)
    session.commit()

    segments = SegmentRepository(session).list_by_source_document(source.id)

    assert registry.requested_source_type == SourceType.DECLARATION
    assert parser.observed_text == "7. I signed the declaration."
    assert source.parser_status == ParserStatus.COMPLETED
    assert source.parser_confidence == 0.77
    assert [segment.rendered_anchor for segment in segments] == ["¶7"]
    assert [segment.raw_text for segment in segments] == ["I signed the declaration."]


def test_ingestion_service_fails_clearly_when_no_parser_is_registered(session: Session) -> None:
    matter = Matter(name="Acme v. Doe", status="ACTIVE")
    session.add(matter)
    session.commit()

    payload = SourceDocumentCreate(
        file_name="exhibit.txt",
        source_type=SourceType.EXHIBIT,
        raw_file_path="/tmp/exhibit.txt",
        content="Exhibit A text.",
    )

    with pytest.raises(ValidationError, match="No parser is registered for source type EXHIBIT"):
        EvidenceIngestionService(session, parser_registry=ParserRegistry()).create_source_with_segments(matter.id, payload)


class SpyRegistry:
    def __init__(self, parser: "SpyParser") -> None:
        self.parser = parser
        self.requested_source_type: SourceType | None = None

    def get(self, source_type: SourceType):
        self.requested_source_type = source_type
        return self.parser


class SpyParser:
    def __init__(
        self,
        *,
        source_type: SourceType,
        segments: list[ParsedSegment],
        confidence: float,
    ) -> None:
        self.source_type = source_type
        self._segments = segments
        self._confidence = confidence
        self.observed_text: str | None = None

    def parse(self, text: str) -> list[ParsedSegment]:
        self.observed_text = text
        return list(self._segments)

    def confidence_for_segments(self, segments: list[ParsedSegment]) -> float:
        assert segments == self._segments
        return self._confidence
