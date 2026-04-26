import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.provenance import build_block_anchor_metadata, build_exhibit_page_anchor_metadata, build_paragraph_anchor_metadata
from app.core.enums import ClaimType, DraftMode, LinkType, ParserStatus, SourceType, SupportStatus
from app.models import Assertion, Base, ClaimUnit, Draft, Matter, Segment, SourceDocument
from app.repositories.links import LinksRepository
from app.repositories.segments import SegmentRepository
from app.schemas.support_link import SupportLinkCreate
from app.services.parsing.normalization import normalize_for_match
from app.services.verification.context_builder import build_claim_context


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    db_session = SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()
        engine.dispose()


def test_links_repository_prevents_duplicate_links(session: Session) -> None:
    claim, source_document = _build_claim_fixture(session)
    segment = _build_segment(
        source_document.id,
        page_start=10,
        line_start=4,
        page_end=10,
        line_end=5,
        speaker="A",
        raw_text="A. Doe signed the contract.",
    )
    session.add(segment)
    session.flush()

    repository = LinksRepository(session)
    payload = SupportLinkCreate(segment_id=segment.id, link_type=LinkType.MANUAL, user_confirmed=True)

    first = repository.create(claim.id, payload)
    second = repository.create(claim.id, payload)

    assert first.id == second.id
    assert first.sequence_order == 1
    assert len(repository.list_by_claim(claim.id)) == 1


def test_build_claim_context_uses_link_order_and_structured_support_items(session: Session) -> None:
    claim, source_document = _build_claim_fixture(session)
    first_segment = _build_segment(
        source_document.id,
        page_start=10,
        line_start=4,
        page_end=10,
        line_end=5,
        speaker="Q",
        raw_text="Q. Did you review the contract?",
        segment_type="QUESTION_BLOCK",
    )
    second_segment = _build_segment(
        source_document.id,
        page_start=10,
        line_start=6,
        page_end=10,
        line_end=8,
        speaker="A",
        raw_text="A. Yes, I reviewed it and signed it.",
        segment_type="ANSWER_BLOCK",
    )
    session.add_all([first_segment, second_segment])
    session.flush()

    links_repository = LinksRepository(session)
    first_link = links_repository.create(
        claim.id,
        SupportLinkCreate(segment_id=second_segment.id, link_type=LinkType.MANUAL, user_confirmed=True),
    )
    second_link = links_repository.create(
        claim.id,
        SupportLinkCreate(segment_id=first_segment.id, link_type=LinkType.MANUAL, user_confirmed=True),
    )

    context = build_claim_context(
        claim,
        links=[second_link, first_link],
        segments=[first_segment, second_segment],
    )

    assert [item.segment_id for item in context.support_items] == [second_segment.id, first_segment.id]
    assert context.citations == ["p.10:6-10:8", "p.10:4-10:5"]
    assert context.support_items[0].speaker == "A"
    assert context.support_items[0].raw_text == "A. Yes, I reviewed it and signed it."
    assert "Anchor: p.10:6-10:8" in context.segment_bundle
    assert "Speaker: A" in context.segment_bundle
    assert "A. Yes, I reviewed it and signed it." in context.segment_bundle


def test_segment_repository_returns_local_context_window(session: Session) -> None:
    _, source_document = _build_claim_fixture(session)
    segments = [
        _build_segment(source_document.id, page_start=11, line_start=1, page_end=11, line_end=1, raw_text="A. One."),
        _build_segment(source_document.id, page_start=11, line_start=2, page_end=11, line_end=2, raw_text="A. Two."),
        _build_segment(source_document.id, page_start=11, line_start=3, page_end=11, line_end=3, raw_text="A. Three."),
        _build_segment(source_document.id, page_start=11, line_start=4, page_end=11, line_end=4, raw_text="A. Four."),
        _build_segment(source_document.id, page_start=11, line_start=5, page_end=11, line_end=5, raw_text="A. Five."),
    ]
    session.add_all(segments)
    session.flush()

    window = SegmentRepository(session).get_local_context_window(segments[2].id, radius=1)

    assert [segment.line_start for segment in window] == [2, 3, 4]
    assert [segment.raw_text for segment in window] == ["A. Two.", "A. Three.", "A. Four."]


def test_build_claim_context_renders_declaration_anchor_metadata(session: Session) -> None:
    claim, source_document = _build_claim_fixture(session, source_type=SourceType.DECLARATION)
    segment = _build_segment(
        source_document.id,
        page_start=None,
        line_start=None,
        page_end=None,
        line_end=None,
        speaker="DECLARANT ¶2",
        raw_text="2. I signed the declaration.",
        segment_type="DECLARATION_PARAGRAPH",
        anchor_metadata=build_paragraph_anchor_metadata(paragraph_number=2, sequence_order=2),
    )
    session.add(segment)
    session.flush()

    link = LinksRepository(session).create(
        claim.id,
        SupportLinkCreate(segment_id=segment.id, link_type=LinkType.MANUAL, user_confirmed=True),
    )

    context = build_claim_context(claim, links=[link], segments=[segment])

    assert context.citations == ["¶2"]
    assert context.support_items[0].anchor == "¶2"
    assert "Anchor: ¶2" in context.segment_bundle
    assert "Speaker: DECLARANT ¶2" in context.segment_bundle


def test_build_claim_context_renders_exhibit_anchor_metadata(session: Session) -> None:
    claim, source_document = _build_claim_fixture(session, source_type=SourceType.EXHIBIT)
    segment = _build_segment(
        source_document.id,
        page_start=None,
        line_start=None,
        page_end=None,
        line_end=None,
        speaker="Subject",
        raw_text="Subject: Contract Status",
        segment_type="EXHIBIT_LABELED_BLOCK",
        anchor_metadata=build_exhibit_page_anchor_metadata(
            page_number=3,
            block_index=1,
            sequence_order=1,
            label="Subject",
        ),
    )
    session.add(segment)
    session.flush()

    link = LinksRepository(session).create(
        claim.id,
        SupportLinkCreate(segment_id=segment.id, link_type=LinkType.MANUAL, user_confirmed=True),
    )

    context = build_claim_context(claim, links=[link], segments=[segment])

    assert context.citations == ["ex. p.3 block 1 [Subject]"]
    assert context.support_items[0].anchor == "ex. p.3 block 1 [Subject]"
    assert "Anchor: ex. p.3 block 1 [Subject]" in context.segment_bundle
    assert "Speaker: Subject" in context.segment_bundle


def test_segment_repository_orders_declaration_segments_by_provenance_metadata(session: Session) -> None:
    _, source_document = _build_claim_fixture(session, source_type=SourceType.DECLARATION)
    segments = [
        _build_segment(
            source_document.id,
            page_start=None,
            line_start=None,
            page_end=None,
            line_end=None,
            raw_text="I, Jane Doe, declare as follows.",
            speaker="DECLARANT",
            segment_type="DECLARATION_BLOCK",
            anchor_metadata=build_block_anchor_metadata(block_index=1, sequence_order=1),
        ),
        _build_segment(
            source_document.id,
            page_start=None,
            line_start=None,
            page_end=None,
            line_end=None,
            raw_text="I signed the declaration.",
            speaker="DECLARANT ¶2",
            segment_type="DECLARATION_PARAGRAPH",
            anchor_metadata=build_paragraph_anchor_metadata(paragraph_number=2, sequence_order=2),
        ),
        _build_segment(
            source_document.id,
            page_start=None,
            line_start=None,
            page_end=None,
            line_end=None,
            raw_text="I declare under penalty of perjury that the foregoing is true.",
            speaker="DECLARANT",
            segment_type="DECLARATION_BLOCK",
            anchor_metadata=build_block_anchor_metadata(block_index=3, sequence_order=3),
        ),
    ]
    session.add_all(segments)
    session.flush()

    ordered_segments = SegmentRepository(session).list_by_source_document(source_document.id)

    assert [segment.rendered_anchor for segment in ordered_segments] == ["block 1", "¶2", "block 3"]
    assert [segment.raw_text for segment in ordered_segments] == [
        "I, Jane Doe, declare as follows.",
        "I signed the declaration.",
        "I declare under penalty of perjury that the foregoing is true.",
    ]


def test_segment_repository_orders_exhibit_segments_by_provenance_metadata(session: Session) -> None:
    _, source_document = _build_claim_fixture(session, source_type=SourceType.EXHIBIT)
    segments = [
        _build_segment(
            source_document.id,
            page_start=None,
            line_start=None,
            page_end=None,
            line_end=None,
            raw_text="Payment received on March 3.",
            speaker=None,
            segment_type="EXHIBIT_PAGE_BLOCK",
            anchor_metadata=build_exhibit_page_anchor_metadata(page_number=1, block_index=2, sequence_order=2),
        ),
        _build_segment(
            source_document.id,
            page_start=None,
            line_start=None,
            page_end=None,
            line_end=None,
            raw_text="Subject: Contract Status",
            speaker="Subject",
            segment_type="EXHIBIT_LABELED_BLOCK",
            anchor_metadata=build_exhibit_page_anchor_metadata(
                page_number=1,
                block_index=1,
                sequence_order=1,
                label="Subject",
            ),
        ),
        _build_segment(
            source_document.id,
            page_start=None,
            line_start=None,
            page_end=None,
            line_end=None,
            raw_text="Follow-up notice sent.",
            speaker=None,
            segment_type="EXHIBIT_PAGE_BLOCK",
            anchor_metadata=build_exhibit_page_anchor_metadata(page_number=2, block_index=1, sequence_order=3),
        ),
    ]
    session.add_all(segments)
    session.flush()

    ordered_segments = SegmentRepository(session).list_by_source_document(source_document.id)

    assert [segment.rendered_anchor for segment in ordered_segments] == [
        "ex. p.1 block 1 [Subject]",
        "ex. p.1 block 2",
        "ex. p.2 block 1",
    ]


def _build_claim_fixture(session: Session, *, source_type: SourceType = SourceType.DEPOSITION) -> tuple[ClaimUnit, SourceDocument]:
    matter = Matter(name="Acme v. Doe", status="ACTIVE")
    source_document = SourceDocument(
        matter=matter,
        file_name="source.txt",
        source_type=source_type,
        raw_file_path="/tmp/source.txt",
        parser_status=ParserStatus.COMPLETED,
        parser_confidence=0.95,
    )
    draft = Draft(matter=matter, title="Motion draft", mode=DraftMode.DRAFT)
    assertion = Assertion(
        draft=draft,
        raw_text="Doe signed the contract.",
        normalized_text=normalize_for_match("Doe signed the contract."),
    )
    claim = ClaimUnit(
        assertion=assertion,
        text="Doe signed the contract.",
        normalized_text=normalize_for_match("Doe signed the contract."),
        claim_type=ClaimType.FACT,
        sequence_order=1,
        support_status=SupportStatus.UNVERIFIED,
    )
    session.add_all([matter, source_document, draft, assertion, claim])
    session.flush()
    return claim, source_document


def _build_segment(
    source_document_id,
    *,
    page_start: int | None,
    line_start: int | None,
    page_end: int | None,
    line_end: int | None,
    raw_text: str,
    speaker: str = "A",
    segment_type: str = "ANSWER_BLOCK",
    anchor_metadata: dict[str, object] | None = None,
) -> Segment:
    return Segment(
        id=uuid.uuid4(),
        source_document_id=source_document_id,
        page_start=page_start,
        line_start=line_start,
        page_end=page_end,
        line_end=line_end,
        anchor_metadata=anchor_metadata,
        raw_text=raw_text,
        normalized_text=normalize_for_match(raw_text),
        speaker=speaker,
        segment_type=segment_type,
    )
