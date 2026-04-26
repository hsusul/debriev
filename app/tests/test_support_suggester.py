import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.provenance import build_paragraph_anchor_metadata
from app.core.enums import ClaimType, DraftMode, ParserStatus, SourceType, SupportStatus
from app.models import Assertion, Base, ClaimUnit, Draft, Matter, Segment, SourceDocument
from app.repositories.evidence_bundles import EvidenceBundleRepository
from app.repositories.segments import SegmentRepository
from app.schemas.evidence_bundle import EvidenceBundleCreate
from app.services.linking.suggester import SupportSuggestionService
from app.services.parsing.normalization import normalize_for_match


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


def test_candidate_segment_retrieval_for_scoped_draft_only_uses_bundle_sources(session: Session) -> None:
    _, _, scoped_source, unscoped_source, draft, _, _, scoped_segments, _ = _build_claim_scope_fixture(session, scoped=True)

    candidates = SegmentRepository(session).list_candidate_segments_for_draft(draft.id)

    assert [segment.source_document_id for segment in candidates] == [scoped_source.id, scoped_source.id]
    assert all(segment.source_document_id != unscoped_source.id for segment in candidates)
    assert [segment.rendered_anchor for segment in candidates] == ["p.10:1-10:1", "p.10:2-10:2"]
    assert [segment.id for segment in candidates] == [scoped_segments[0].id, scoped_segments[1].id]


def test_candidate_segment_retrieval_for_unscoped_draft_falls_back_to_all_matter_sources(session: Session) -> None:
    _, _, scoped_source, unscoped_source, draft, _, _, scoped_segments, unscoped_segments = _build_claim_scope_fixture(
        session,
        scoped=False,
    )

    candidates = SegmentRepository(session).list_candidate_segments_for_draft(draft.id)

    assert [segment.source_document_id for segment in candidates] == [
        scoped_source.id,
        scoped_source.id,
        unscoped_source.id,
        unscoped_source.id,
    ]
    assert [segment.rendered_anchor for segment in candidates] == [
        "p.10:1-10:1",
        "p.10:2-10:2",
        "¶1",
        "¶2",
    ]
    assert [segment.id for segment in candidates] == [
        scoped_segments[0].id,
        scoped_segments[1].id,
        unscoped_segments[0].id,
        unscoped_segments[1].id,
    ]


def test_support_suggestion_for_claim_respects_draft_scope(session: Session) -> None:
    _, _, _, _, _, claim, _, scoped_segments, unscoped_segments = _build_claim_scope_fixture(session, scoped=True)

    suggestions = SupportSuggestionService(session).suggest_for_claim(claim.id)

    assert [suggestion.segment_id for suggestion in suggestions] == [scoped_segments[0].id]
    assert all(suggestion.segment_id != segment.id for segment in unscoped_segments for suggestion in suggestions)


def test_support_suggestion_for_unscoped_claim_falls_back_to_all_matter_sources(session: Session) -> None:
    _, _, _, _, _, claim, _, scoped_segments, unscoped_segments = _build_claim_scope_fixture(session, scoped=False)

    suggestions = SupportSuggestionService(session).suggest_for_claim(claim.id)

    assert [suggestion.segment_id for suggestion in suggestions] == [
        unscoped_segments[0].id,
        scoped_segments[0].id,
    ]
    assert suggestions[0].score > suggestions[1].score


def _build_claim_scope_fixture(session: Session, *, scoped: bool):
    matter = Matter(name="Acme v. Doe", status="ACTIVE")
    scoped_source = SourceDocument(
        matter=matter,
        file_name="a-deposition.txt",
        source_type=SourceType.DEPOSITION,
        raw_file_path="/tmp/a-deposition.txt",
        parser_status=ParserStatus.COMPLETED,
        parser_confidence=0.95,
    )
    unscoped_source = SourceDocument(
        matter=matter,
        file_name="b-declaration.txt",
        source_type=SourceType.DECLARATION,
        raw_file_path="/tmp/b-declaration.txt",
        parser_status=ParserStatus.COMPLETED,
        parser_confidence=0.85,
    )
    session.add_all([matter, scoped_source, unscoped_source])
    session.flush()

    if scoped:
        bundle = EvidenceBundleRepository(session).create(
            matter.id,
            EvidenceBundleCreate(
                name="Scoped Record",
                source_document_ids=[scoped_source.id],
            ),
        )
        draft = Draft(
            matter=matter,
            evidence_bundle_id=bundle.id,
            title="Scoped draft",
            mode=DraftMode.COMPILE,
        )
    else:
        bundle = None
        draft = Draft(
            matter=matter,
            title="Unscoped draft",
            mode=DraftMode.COMPILE,
        )

    assertion = Assertion(
        draft=draft,
        paragraph_index=1,
        sentence_index=1,
        raw_text="Doe signed the agreement.",
        normalized_text=normalize_for_match("Doe signed the agreement."),
    )
    claim = ClaimUnit(
        assertion=assertion,
        text="Doe signed the agreement.",
        normalized_text=normalize_for_match("Doe signed the agreement."),
        claim_type=ClaimType.FACT,
        sequence_order=1,
        support_status=SupportStatus.UNVERIFIED,
    )
    session.add_all([draft, assertion, claim])
    session.flush()

    scoped_segments = [
        Segment(
            source_document_id=scoped_source.id,
            page_start=10,
            line_start=1,
            page_end=10,
            line_end=1,
            raw_text="A. Doe approved the declaration.",
            normalized_text=normalize_for_match("A. Doe approved the declaration."),
            speaker="A",
            segment_type="ANSWER_BLOCK",
        ),
        Segment(
            source_document_id=scoped_source.id,
            page_start=10,
            line_start=2,
            page_end=10,
            line_end=2,
            raw_text="A. Counsel requested a recess.",
            normalized_text=normalize_for_match("A. Counsel requested a recess."),
            speaker="A",
            segment_type="ANSWER_BLOCK",
        ),
    ]
    unscoped_segments = [
        Segment(
            source_document_id=unscoped_source.id,
            page_start=None,
            line_start=None,
            page_end=None,
            line_end=None,
            anchor_metadata=build_paragraph_anchor_metadata(paragraph_number=1, sequence_order=1),
            raw_text="I signed the agreement.",
            normalized_text=normalize_for_match("I signed the agreement."),
            speaker="DECLARANT ¶1",
            segment_type="DECLARATION_PARAGRAPH",
        ),
        Segment(
            source_document_id=unscoped_source.id,
            page_start=None,
            line_start=None,
            page_end=None,
            line_end=None,
            anchor_metadata=build_paragraph_anchor_metadata(paragraph_number=2, sequence_order=2),
            raw_text="I reviewed billing records.",
            normalized_text=normalize_for_match("I reviewed billing records."),
            speaker="DECLARANT ¶2",
            segment_type="DECLARATION_PARAGRAPH",
        ),
    ]
    session.add_all(scoped_segments + unscoped_segments)
    session.commit()

    return matter, bundle, scoped_source, unscoped_source, draft, claim, assertion, scoped_segments, unscoped_segments
