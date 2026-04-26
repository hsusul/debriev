import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.enums import ClaimType, DraftMode, LinkType, ParserStatus, SourceType, SupportStatus
from app.models import Assertion, Base, ClaimUnit, Draft, Matter, Segment, SourceDocument, SupportLink
from app.repositories.evidence_bundles import EvidenceBundleRepository
from app.schemas.evidence_bundle import EvidenceBundleCreate
from app.services.linking.validation import LinkValidationService
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


def test_revalidate_existing_link_accepts_valid_in_scope_link(session: Session) -> None:
    link = _seed_link(session, scoped=True, link_target="in_scope")

    result = LinkValidationService(session).revalidate_link(link.id, link.claim_unit_id, link.segment_id)

    assert result.is_valid is True
    assert result.code is None
    assert result.message is None


def test_revalidate_existing_link_detects_out_of_scope_segment(session: Session) -> None:
    link = _seed_link(session, scoped=True, link_target="out_of_scope")

    result = LinkValidationService(session).revalidate_link(link.id, link.claim_unit_id, link.segment_id)

    assert result.is_valid is False
    assert result.code == "out_of_scope_support_link"
    assert result.message == "Segment source document is outside the draft evidence bundle."


def test_revalidate_existing_link_detects_cross_matter_segment(session: Session) -> None:
    link = _seed_link(session, scoped=False, link_target="cross_matter")

    result = LinkValidationService(session).revalidate_link(link.id, link.claim_unit_id, link.segment_id)

    assert result.is_valid is False
    assert result.code == "cross_matter_support_link"
    assert result.message == "Claim and segment must belong to the same matter."


def _seed_link(session: Session, *, scoped: bool, link_target: str) -> SupportLink:
    matter = Matter(name="Acme v. Doe", status="ACTIVE")
    in_scope_source = SourceDocument(
        matter=matter,
        file_name="in-scope.txt",
        source_type=SourceType.DEPOSITION,
        raw_file_path="/tmp/in-scope.txt",
        parser_status=ParserStatus.COMPLETED,
        parser_confidence=0.95,
    )
    out_of_scope_source = SourceDocument(
        matter=matter,
        file_name="out-of-scope.txt",
        source_type=SourceType.EXHIBIT,
        raw_file_path="/tmp/out-of-scope.txt",
        parser_status=ParserStatus.COMPLETED,
        parser_confidence=0.8,
    )
    other_matter = Matter(name="Other Matter", status="ACTIVE")
    cross_matter_source = SourceDocument(
        matter=other_matter,
        file_name="cross-matter.txt",
        source_type=SourceType.DECLARATION,
        raw_file_path="/tmp/cross-matter.txt",
        parser_status=ParserStatus.COMPLETED,
        parser_confidence=0.85,
    )
    session.add_all([matter, in_scope_source, out_of_scope_source, other_matter, cross_matter_source])
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
        title="Draft",
        mode=DraftMode.DRAFT,
    )
    assertion = Assertion(
        draft=draft,
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

    in_scope_segment = _build_segment(in_scope_source.id, "A. Doe signed the agreement.")
    out_of_scope_segment = _build_segment(out_of_scope_source.id, "Document reflects the signed agreement.")
    cross_matter_segment = _build_segment(cross_matter_source.id, "I signed the declaration.")
    session.add_all([in_scope_segment, out_of_scope_segment, cross_matter_segment])
    session.flush()

    segment_by_target = {
        "in_scope": in_scope_segment,
        "out_of_scope": out_of_scope_segment,
        "cross_matter": cross_matter_segment,
    }
    link = SupportLink(
        id=uuid.uuid4(),
        claim_unit_id=claim.id,
        segment_id=segment_by_target[link_target].id,
        sequence_order=1,
        link_type=LinkType.MANUAL,
        user_confirmed=True,
    )
    session.add(link)
    session.commit()
    return link


def _build_segment(source_document_id: uuid.UUID, raw_text: str) -> Segment:
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
