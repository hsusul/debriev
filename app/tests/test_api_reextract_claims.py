import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.deps import get_db_session
from app.core.enums import ClaimType, DraftMode, LinkType, ParserStatus, SourceType, SupportStatus
from app.main import create_app
from app.models import Assertion, Base, ClaimUnit, Draft, Matter, Segment, SourceDocument, SupportLink
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


@pytest.fixture
def client(session: Session) -> TestClient:
    app = create_app()

    def override_get_db():
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[get_db_session] = override_get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_compare_endpoint_surfaces_legacy_unversioned_assertion(client: TestClient, session: Session) -> None:
    assertion = _build_assertion(session, raw_text="Doe signed the contract and approved the invoice.")
    _add_claim(
        session,
        assertion=assertion,
        text="Doe signed the contract and approved the invoice.",
        sequence_order=1,
    )
    session.commit()

    response = client.post(f"/api/v1/assertions/{assertion.id}/reextract/compare", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["assertion_id"] == str(assertion.id)
    assert payload["existing_metadata"] == {
        "status": "legacy_unversioned",
        "strategy": None,
        "version": None,
        "snapshot_present": False,
    }
    assert payload["proposed_metadata"] == {
        "strategy": "structured",
        "version": CURRENT_EXTRACTION_VERSION,
    }
    assert [claim["text"] for claim in payload["existing_claims"]] == [
        "Doe signed the contract and approved the invoice.",
    ]
    assert [claim["text"] for claim in payload["proposed_claims"]] == [
        "Doe signed the contract",
        "Doe approved the invoice.",
    ]
    assert payload["materially_changed"] is True
    assert payload["apply_requires_replacement"] is True
    assert payload["can_apply"] is True
    assert payload["blocked_reasons"] == []


def test_compare_endpoint_surfaces_unchanged_versioned_assertion(client: TestClient, session: Session) -> None:
    assertion = _build_assertion(session, raw_text="Doe signed the contract and approved the invoice.")
    ClaimExtractionService(session, extraction_mode="structured").extract_from_assertion(assertion.id)
    session.commit()

    response = client.post(f"/api/v1/assertions/{assertion.id}/reextract/compare", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["existing_metadata"]["status"] == "versioned"
    assert payload["existing_metadata"]["strategy"] == "structured"
    assert payload["existing_metadata"]["version"] == CURRENT_EXTRACTION_VERSION
    assert payload["existing_metadata"]["snapshot_present"] is True
    assert payload["proposed_metadata"] == {
        "strategy": "structured",
        "version": CURRENT_EXTRACTION_VERSION,
    }
    assert payload["materially_changed"] is False
    assert payload["apply_requires_replacement"] is False
    assert payload["can_apply"] is True
    assert payload["blocked_reasons"] == []


def test_compare_endpoint_surfaces_blocked_reasons_for_material_change(
    client: TestClient,
    session: Session,
) -> None:
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

    response = client.post(f"/api/v1/assertions/{assertion.id}/reextract/compare", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["materially_changed"] is True
    assert payload["apply_requires_replacement"] is True
    assert payload["can_apply"] is False
    assert payload["blocked_reasons"] == [
        "Re-extraction cannot replace persisted claim units while support links are attached."
    ]


def test_apply_endpoint_replaces_claims_and_updates_metadata_for_legacy_assertion(
    client: TestClient,
    session: Session,
) -> None:
    assertion = _build_assertion(session, raw_text="Doe signed the contract and approved the invoice.")
    existing_claim = _add_claim(
        session,
        assertion=assertion,
        text="Doe signed the contract and approved the invoice.",
        sequence_order=1,
    )
    session.commit()

    response = client.post(f"/api/v1/assertions/{assertion.id}/reextract/apply", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["assertion_id"] == str(assertion.id)
    assert payload["applied_strategy"] == "structured"
    assert payload["applied_version"] == CURRENT_EXTRACTION_VERSION
    assert payload["materially_changed"] is True
    assert payload["apply_requires_replacement"] is True
    assert payload["claims_replaced"] is True
    assert payload["metadata_updated"] is True
    assert [claim["text"] for claim in payload["resulting_claims"]] == [
        "Doe signed the contract",
        "Doe approved the invoice.",
    ]
    assert str(existing_claim.id) not in [claim["claim_id"] for claim in payload["resulting_claims"]]
    assert payload["updated_metadata"] == {
        "status": "versioned",
        "strategy": "structured",
        "version": CURRENT_EXTRACTION_VERSION,
        "snapshot_present": True,
    }
    assert payload["snapshot_summary"] == {
        "present": True,
        "claim_count": 2,
    }
    assert payload["notes"] == [
        "Extraction metadata and snapshot were refreshed.",
        "Persisted claim units were replaced during re-extraction.",
    ]

    session.refresh(assertion)
    assert assertion.extraction_strategy == "structured"
    assert assertion.extraction_version == CURRENT_EXTRACTION_VERSION
    assert assertion.extraction_snapshot is not None


def test_apply_endpoint_updates_metadata_without_replacing_when_claims_are_unchanged(
    client: TestClient,
    session: Session,
) -> None:
    assertion = _build_assertion(session, raw_text="Doe signed the contract.")
    existing_claim = _add_claim(
        session,
        assertion=assertion,
        text="Doe signed the contract.",
        sequence_order=1,
    )
    session.commit()

    response = client.post(f"/api/v1/assertions/{assertion.id}/reextract/apply", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["materially_changed"] is False
    assert payload["apply_requires_replacement"] is False
    assert payload["claims_replaced"] is False
    assert payload["metadata_updated"] is True
    assert [claim["claim_id"] for claim in payload["resulting_claims"]] == [str(existing_claim.id)]
    assert payload["updated_metadata"] == {
        "status": "versioned",
        "strategy": "structured",
        "version": CURRENT_EXTRACTION_VERSION,
        "snapshot_present": True,
    }
    assert payload["snapshot_summary"] == {
        "present": True,
        "claim_count": 1,
    }
    assert payload["notes"] == [
        "Extraction metadata and snapshot were refreshed.",
        "Existing persisted claim units already matched the selected extraction output.",
    ]

    session.refresh(assertion)
    assert assertion.extraction_strategy == "structured"
    assert assertion.extraction_version == CURRENT_EXTRACTION_VERSION
    assert assertion.extraction_snapshot is not None
    assert assertion.extraction_snapshot["claims"][0]["claim_id"] == str(existing_claim.id)


def test_apply_endpoint_fails_clearly_when_replacement_is_blocked(
    client: TestClient,
    session: Session,
) -> None:
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

    response = client.post(f"/api/v1/assertions/{assertion.id}/reextract/apply", json={})

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Re-extraction cannot replace persisted claim units while support links are attached."
    }


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
