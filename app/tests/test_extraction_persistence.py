import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.enums import ClaimType, DraftMode, SupportStatus
from app.models import Assertion, Base, ClaimUnit, Draft, Matter
from app.services.claims.extractor import CURRENT_EXTRACTION_VERSION, ClaimExtractionService
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


def test_extract_from_assertion_persists_structured_version_and_snapshot(session: Session) -> None:
    assertion = _build_assertion(
        session,
        raw_text="Subject: Contract Status. Doe signed the agreement and later emailed counsel regarding payment.",
    )

    claims = ClaimExtractionService(session, extraction_mode="structured").extract_from_assertion(assertion.id)
    session.refresh(assertion)

    assert assertion.extraction_strategy == "structured"
    assert assertion.extraction_version == CURRENT_EXTRACTION_VERSION
    assert assertion.extraction_snapshot is not None
    assert assertion.extraction_snapshot["assertion_id"] == str(assertion.id)
    assert assertion.extraction_snapshot["source_assertion_text"] == assertion.raw_text
    assert assertion.extraction_snapshot["normalized_assertion_text"] == assertion.normalized_text
    assert assertion.extraction_snapshot["extraction_strategy"] == "structured"
    assert assertion.extraction_snapshot["extraction_version"] == CURRENT_EXTRACTION_VERSION
    assert assertion.extraction_snapshot["claims"] == [
        {
            "claim_id": str(claims[0].id),
            "text": "Doe signed the agreement",
            "normalized_text": normalize_for_match("Doe signed the agreement"),
            "claim_type": ClaimType.FACT.value,
            "sequence_order": 1,
        },
        {
            "claim_id": str(claims[1].id),
            "text": "Doe later emailed counsel regarding payment.",
            "normalized_text": normalize_for_match("Doe later emailed counsel regarding payment."),
            "claim_type": ClaimType.FACT.value,
            "sequence_order": 2,
        },
    ]


def test_extract_from_assertion_persists_legacy_strategy_when_requested(session: Session) -> None:
    assertion = _build_assertion(
        session,
        raw_text="Doe signed the contract and approved the invoice.",
    )

    claims = ClaimExtractionService(session, extraction_mode="legacy").extract_from_assertion(assertion.id)
    session.refresh(assertion)

    assert [claim.text for claim in claims] == [
        "Doe signed the contract",
        "Doe approved the invoice.",
    ]
    assert assertion.extraction_strategy == "legacy"
    assert assertion.extraction_version == CURRENT_EXTRACTION_VERSION
    assert assertion.extraction_snapshot is not None
    assert [entry["text"] for entry in assertion.extraction_snapshot["claims"]] == [
        "Doe signed the contract",
        "Doe approved the invoice.",
    ]


def test_existing_unversioned_assertion_with_claims_remains_readable(session: Session) -> None:
    assertion = _build_assertion(
        session,
        raw_text="Doe signed the contract.",
    )
    existing_claim = ClaimUnit(
        assertion_id=assertion.id,
        text="Doe signed the contract.",
        normalized_text=normalize_for_match("Doe signed the contract."),
        claim_type=ClaimType.FACT,
        sequence_order=1,
        support_status=SupportStatus.UNVERIFIED,
    )
    session.add(existing_claim)
    session.commit()

    claims = ClaimExtractionService(session).extract_from_assertion(assertion.id)
    session.refresh(assertion)

    assert [claim.id for claim in claims] == [existing_claim.id]
    assert assertion.extraction_strategy is None
    assert assertion.extraction_version is None
    assert assertion.extraction_snapshot is None


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
