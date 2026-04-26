import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.enums import ClaimType, DraftMode, LinkType, ParserStatus, SourceType, SupportStatus
from app.core.exceptions import ConflictError
from app.models import Assertion, Base, ClaimUnit, Draft, Matter, Segment, SourceDocument, SupportLink
from app.services.claims.extractor import CURRENT_EXTRACTION_VERSION, ClaimExtractionService
from app.services.parsing.normalization import normalize_for_match
from app.services.workflows.reextract_claims import ClaimReExtractionService


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


def test_reextract_compare_surfaces_legacy_unversioned_assertion(session: Session) -> None:
    assertion = _build_assertion(session, raw_text="Doe signed the contract and approved the invoice.")
    _add_claim(
        session,
        assertion=assertion,
        text="Doe signed the contract and approved the invoice.",
        sequence_order=1,
    )
    session.commit()

    comparison = ClaimReExtractionService(session).compare_assertion(assertion.id, mode="structured")

    assert comparison.existing_metadata.status == "legacy_unversioned"
    assert comparison.existing_metadata.strategy is None
    assert comparison.proposed_metadata.strategy == "structured"
    assert comparison.proposed_metadata.version == CURRENT_EXTRACTION_VERSION
    assert [claim.text for claim in comparison.existing_claims] == [
        "Doe signed the contract and approved the invoice.",
    ]
    assert [claim.text for claim in comparison.proposed_claims] == [
        "Doe signed the contract",
        "Doe approved the invoice.",
    ]
    assert comparison.materially_changed is True
    assert comparison.apply_requires_replacement is True
    assert comparison.can_apply is True


def test_reextract_compare_detects_no_change_for_versioned_assertion(session: Session) -> None:
    assertion = _build_assertion(session, raw_text="Doe signed the contract and approved the invoice.")
    ClaimExtractionService(session, extraction_mode="structured").extract_from_assertion(assertion.id)
    session.commit()

    comparison = ClaimReExtractionService(session).compare_assertion(assertion.id, mode="structured")

    assert comparison.existing_metadata.status == "versioned"
    assert comparison.existing_metadata.strategy == "structured"
    assert comparison.existing_metadata.version == CURRENT_EXTRACTION_VERSION
    assert comparison.materially_changed is False
    assert comparison.apply_requires_replacement is False
    assert comparison.can_apply is True
    assert [claim.text for claim in comparison.existing_claims] == [
        "Doe signed the contract",
        "Doe approved the invoice.",
    ]
    assert [claim.text for claim in comparison.proposed_claims] == [
        "Doe signed the contract",
        "Doe approved the invoice.",
    ]


def test_reextract_apply_replaces_claims_and_updates_metadata(session: Session) -> None:
    assertion = _build_assertion(session, raw_text="Doe signed the contract and approved the invoice.")
    existing_claim = _add_claim(
        session,
        assertion=assertion,
        text="Doe signed the contract and approved the invoice.",
        sequence_order=1,
    )
    session.commit()

    result = ClaimReExtractionService(session).apply_assertion(assertion.id, mode="structured")
    session.refresh(assertion)

    assert result.metadata_updated is True
    assert result.claims_replaced is True
    assert [claim.text for claim in result.claims] == [
        "Doe signed the contract",
        "Doe approved the invoice.",
    ]
    assert existing_claim.id not in [claim.id for claim in result.claims]
    assert assertion.extraction_strategy == "structured"
    assert assertion.extraction_version == CURRENT_EXTRACTION_VERSION
    assert assertion.extraction_snapshot is not None
    assert [entry["text"] for entry in assertion.extraction_snapshot["claims"]] == [
        "Doe signed the contract",
        "Doe approved the invoice.",
    ]


def test_reextract_apply_updates_metadata_without_replacing_when_claims_are_unchanged(session: Session) -> None:
    assertion = _build_assertion(session, raw_text="Doe signed the contract.")
    existing_claim = _add_claim(
        session,
        assertion=assertion,
        text="Doe signed the contract.",
        sequence_order=1,
    )
    session.commit()

    result = ClaimReExtractionService(session).apply_assertion(assertion.id, mode="structured")
    session.refresh(assertion)

    assert result.metadata_updated is True
    assert result.claims_replaced is False
    assert [claim.id for claim in result.claims] == [existing_claim.id]
    assert assertion.extraction_strategy == "structured"
    assert assertion.extraction_version == CURRENT_EXTRACTION_VERSION
    assert assertion.extraction_snapshot is not None
    assert assertion.extraction_snapshot["claims"][0]["claim_id"] == str(existing_claim.id)


def test_reextract_apply_blocks_when_existing_claim_has_support_links(session: Session) -> None:
    matter = Matter(name="Acme v. Doe", status="ACTIVE")
    source_document = SourceDocument(
        matter=matter,
        file_name="deposition.txt",
        source_type=SourceType.DEPOSITION,
        raw_file_path="/tmp/deposition.txt",
        parser_status=ParserStatus.COMPLETED,
        parser_confidence=0.95,
    )
    draft = Draft(matter=matter, title="Motion draft", mode=DraftMode.DRAFT)
    assertion = Assertion(
        draft=draft,
        raw_text="Doe signed the contract and approved the invoice.",
        normalized_text=normalize_for_match("Doe signed the contract and approved the invoice."),
    )
    claim = ClaimUnit(
        assertion=assertion,
        text="Doe signed the contract and approved the invoice.",
        normalized_text=normalize_for_match("Doe signed the contract and approved the invoice."),
        claim_type=ClaimType.FACT,
        sequence_order=1,
        support_status=SupportStatus.UNVERIFIED,
    )
    segment = Segment(
        id=uuid.uuid4(),
        source_document=source_document,
        page_start=10,
        line_start=1,
        page_end=10,
        line_end=2,
        raw_text="A. Doe signed the contract.",
        normalized_text=normalize_for_match("A. Doe signed the contract."),
        speaker="A",
        segment_type="ANSWER_BLOCK",
    )
    link = SupportLink(
        claim_unit=claim,
        segment=segment,
        sequence_order=1,
        link_type=LinkType.MANUAL,
        citation_text=None,
        user_confirmed=True,
    )
    session.add_all([matter, source_document, draft, assertion, claim, segment, link])
    session.commit()

    comparison = ClaimReExtractionService(session).compare_assertion(assertion.id, mode="structured")

    assert comparison.materially_changed is True
    assert comparison.apply_requires_replacement is True
    assert comparison.can_apply is False
    assert comparison.blocked_reasons == [
        "Re-extraction cannot replace persisted claim units while support links are attached."
    ]

    with pytest.raises(
        ConflictError,
        match="cannot replace persisted claim units while support links are attached",
    ):
        ClaimReExtractionService(session).apply_assertion(assertion.id, mode="structured")


def _build_assertion(session: Session, *, raw_text: str) -> Assertion:
    matter = Matter(name="Acme v. Doe", status="ACTIVE")
    draft = Draft(matter=matter, title="Motion draft", mode=DraftMode.DRAFT)
    assertion = Assertion(
        id=uuid.uuid4(),
        draft=draft,
        raw_text=raw_text,
        normalized_text=normalize_for_match(raw_text),
    )
    session.add_all([matter, draft, assertion])
    session.commit()
    return assertion


def _add_claim(session: Session, *, assertion: Assertion, text: str, sequence_order: int) -> ClaimUnit:
    claim = ClaimUnit(
        id=uuid.uuid4(),
        assertion_id=assertion.id,
        text=text,
        normalized_text=normalize_for_match(text),
        claim_type=ClaimType.FACT,
        sequence_order=sequence_order,
        support_status=SupportStatus.UNVERIFIED,
    )
    session.add(claim)
    session.flush()
    return claim
