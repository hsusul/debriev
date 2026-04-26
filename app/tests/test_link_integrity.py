import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.enums import ClaimType, DraftMode, LinkType, ParserStatus, SourceType, SupportStatus
from app.core.exceptions import ValidationError
from app.models import Assertion, Base, ClaimUnit, Draft, Matter, Segment, SourceDocument, SupportLink
from app.repositories.evidence_bundles import EvidenceBundleRepository
from app.repositories.links import LinksRepository
from app.schemas.evidence_bundle import EvidenceBundleCreate
from app.schemas.support_link import SupportLinkCreate
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


def test_in_scope_same_matter_link_creation_succeeds(session: Session) -> None:
    claim, in_scope_source, _, _ = _build_claim_with_sources(session, scoped=True)
    segment = _build_segment(in_scope_source.id, raw_text="A. Doe signed the agreement.")
    session.add(segment)
    session.flush()

    link = LinksRepository(session).create(
        claim.id,
        SupportLinkCreate(segment_id=segment.id, link_type=LinkType.MANUAL, user_confirmed=True),
    )

    assert link.claim_unit_id == claim.id
    assert link.segment_id == segment.id
    assert link.sequence_order == 1


def test_out_of_scope_link_creation_fails_for_scoped_draft(session: Session) -> None:
    claim, _, excluded_source, _ = _build_claim_with_sources(session, scoped=True)
    segment = _build_segment(excluded_source.id, raw_text="A. Doe signed the agreement.")
    session.add(segment)
    session.flush()

    with pytest.raises(ValidationError, match="outside the draft evidence bundle"):
        LinksRepository(session).create(
            claim.id,
            SupportLinkCreate(segment_id=segment.id, link_type=LinkType.MANUAL, user_confirmed=True),
        )


def test_cross_matter_link_creation_fails(session: Session) -> None:
    claim, in_scope_source, _, _ = _build_claim_with_sources(session, scoped=False)
    other_matter = Matter(name="Other Matter", status="ACTIVE")
    other_source = SourceDocument(
        matter=other_matter,
        file_name="other-source.txt",
        source_type=SourceType.DECLARATION,
        raw_file_path="/tmp/other-source.txt",
        parser_status=ParserStatus.COMPLETED,
        parser_confidence=0.9,
    )
    del in_scope_source
    session.add_all([other_matter, other_source])
    session.flush()

    segment = _build_segment(other_source.id, raw_text="I signed the declaration.")
    session.add(segment)
    session.flush()

    with pytest.raises(ValidationError, match="must belong to the same matter"):
        LinksRepository(session).create(
            claim.id,
            SupportLinkCreate(segment_id=segment.id, link_type=LinkType.MANUAL, user_confirmed=True),
        )


def test_unscoped_draft_preserves_same_matter_fallback(session: Session) -> None:
    claim, _, fallback_source, _ = _build_claim_with_sources(session, scoped=False)
    segment = _build_segment(fallback_source.id, raw_text="A. Doe delivered the notice.")
    session.add(segment)
    session.flush()

    link = LinksRepository(session).create(
        claim.id,
        SupportLinkCreate(segment_id=segment.id, link_type=LinkType.MANUAL, user_confirmed=True),
    )

    assert link.segment_id == segment.id


def test_support_link_unique_constraint_blocks_duplicate_rows(session: Session) -> None:
    claim, in_scope_source, _, _ = _build_claim_with_sources(session, scoped=False)
    segment = _build_segment(in_scope_source.id, raw_text="A. Doe signed the agreement.")
    session.add(segment)
    session.flush()

    repository = LinksRepository(session)
    repository.create(
        claim.id,
        SupportLinkCreate(segment_id=segment.id, link_type=LinkType.MANUAL, user_confirmed=True),
    )
    session.flush()

    duplicate = SupportLink(
        claim_unit_id=claim.id,
        segment_id=segment.id,
        sequence_order=2,
        link_type=LinkType.MANUAL,
        user_confirmed=True,
    )
    session.add(duplicate)

    with pytest.raises(IntegrityError):
        session.flush()

    session.rollback()


def _build_claim_with_sources(
    session: Session,
    *,
    scoped: bool,
) -> tuple[ClaimUnit, SourceDocument, SourceDocument, Draft]:
    matter = Matter(name="Acme v. Doe", status="ACTIVE")
    in_scope_source = SourceDocument(
        matter=matter,
        file_name="in-scope.txt",
        source_type=SourceType.DEPOSITION,
        raw_file_path="/tmp/in-scope.txt",
        parser_status=ParserStatus.COMPLETED,
        parser_confidence=0.95,
    )
    fallback_source = SourceDocument(
        matter=matter,
        file_name="fallback.txt",
        source_type=SourceType.EXHIBIT,
        raw_file_path="/tmp/fallback.txt",
        parser_status=ParserStatus.COMPLETED,
        parser_confidence=0.8,
    )
    session.add_all([matter, in_scope_source, fallback_source])
    session.flush()

    evidence_bundle_id = None
    if scoped:
        evidence_bundle = EvidenceBundleRepository(session).create(
            matter.id,
            EvidenceBundleCreate(
                name="Scoped Record",
                source_document_ids=[in_scope_source.id],
            ),
        )
        evidence_bundle_id = evidence_bundle.id

    draft = Draft(
        matter=matter,
        evidence_bundle_id=evidence_bundle_id,
        title="Motion draft",
        mode=DraftMode.DRAFT,
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

    return claim, in_scope_source, fallback_source, draft


def _build_segment(source_document_id: uuid.UUID, *, raw_text: str) -> Segment:
    return Segment(
        id=uuid.uuid4(),
        source_document_id=source_document_id,
        page_start=10,
        line_start=4,
        page_end=10,
        line_end=5,
        raw_text=raw_text,
        normalized_text=normalize_for_match(raw_text),
        speaker="A",
        segment_type="ANSWER_BLOCK",
    )
