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


def test_draft_reextract_preview_endpoint_surfaces_mixed_batch_statuses(
    client: TestClient,
    session: Session,
) -> None:
    workspace = _seed_batch_reextract_draft(session)

    response = client.post(f"/api/v1/drafts/{workspace['draft'].id}/reextract/preview", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"]
    assert payload["draft_id"] == str(workspace["draft"].id)
    assert payload["requested_mode"] == "structured"
    assert payload["total_assertions"] == 4
    assert payload["ready_assertions"] == 2
    assert payload["unchanged_assertions"] == 1
    assert payload["blocked_assertions"] == 1
    assert payload["materially_changed_assertions"] == 2
    assert payload["legacy_unversioned_assertions"] == 3
    assert [item["status"] for item in payload["items"]] == [
        "ready",
        "ready",
        "unchanged",
        "blocked",
    ]
    assert payload["items"][3]["blocked_reasons"] == [
        "Re-extraction cannot replace persisted claim units while support links are attached."
    ]


def test_draft_reextract_apply_endpoint_only_applies_safe_assertions(
    client: TestClient,
    session: Session,
) -> None:
    workspace = _seed_batch_reextract_draft(session)

    response = client.post(f"/api/v1/drafts/{workspace['draft'].id}/reextract/apply", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"]
    assert payload["draft_id"] == str(workspace["draft"].id)
    assert payload["requested_mode"] == "structured"
    assert payload["total_assertions"] == 4
    assert payload["applied_assertions"] == 2
    assert payload["skipped_assertions"] == 1
    assert payload["blocked_assertions"] == 1
    assert payload["replaced_assertions"] == 1
    assert payload["metadata_only_assertions"] == 1
    assert [item["status"] for item in payload["items"]] == [
        "applied",
        "applied",
        "skipped",
        "blocked",
    ]

    replaced_item = payload["items"][0]
    assert replaced_item["claims_replaced"] is True
    assert replaced_item["metadata_updated"] is True
    assert [claim["text"] for claim in replaced_item["resulting_claims"]] == [
        "Doe signed the contract",
        "Doe approved the invoice.",
    ]

    metadata_only_item = payload["items"][1]
    assert metadata_only_item["claims_replaced"] is False
    assert metadata_only_item["metadata_updated"] is True
    assert [claim["claim_id"] for claim in metadata_only_item["resulting_claims"]] == [
        str(workspace["legacy_unchanged"]["claim"].id)
    ]

    skipped_item = payload["items"][2]
    assert skipped_item["claims_replaced"] is False
    assert skipped_item["metadata_updated"] is False
    assert skipped_item["notes"] == [
        "Assertion already matches the selected extraction strategy/version and needs no migration."
    ]

    blocked_item = payload["items"][3]
    assert blocked_item["status"] == "blocked"
    assert blocked_item["blocked_reasons"] == [
        "Re-extraction cannot replace persisted claim units while support links are attached."
    ]

    session.refresh(workspace["blocked_changed"]["assertion"])
    assert workspace["blocked_changed"]["assertion"].extraction_strategy is None
    assert workspace["blocked_changed"]["assertion"].extraction_version is None
    assert workspace["blocked_changed"]["assertion"].extraction_snapshot is None


def _seed_batch_reextract_draft(session: Session) -> dict[str, object]:
    matter = Matter(name="Acme v. Doe", status="ACTIVE")
    source_document = SourceDocument(
        matter=matter,
        file_name="deposition.txt",
        source_type=SourceType.DEPOSITION,
        raw_file_path="/tmp/deposition.txt",
        parser_status=ParserStatus.COMPLETED,
        parser_confidence=0.95,
    )
    draft = Draft(matter=matter, title="Migration draft", mode=DraftMode.DRAFT)
    session.add_all([matter, source_document, draft])
    session.flush()

    legacy_changed_assertion = _build_assertion(
        session,
        draft=draft,
        raw_text="Doe signed the contract and approved the invoice.",
        paragraph_index=1,
    )
    _add_claim(
        session,
        assertion=legacy_changed_assertion,
        text="Doe signed the contract and approved the invoice.",
        sequence_order=1,
    )

    legacy_unchanged_assertion = _build_assertion(
        session,
        draft=draft,
        raw_text="Doe signed the contract.",
        paragraph_index=2,
    )
    legacy_unchanged_claim = _add_claim(
        session,
        assertion=legacy_unchanged_assertion,
        text="Doe signed the contract.",
        sequence_order=1,
    )

    versioned_current_assertion = _build_assertion(
        session,
        draft=draft,
        raw_text="Doe emailed counsel.",
        paragraph_index=3,
    )
    ClaimExtractionService(session, extraction_mode="structured").extract_from_assertion(versioned_current_assertion.id)

    blocked_changed_assertion = _build_assertion(
        session,
        draft=draft,
        raw_text="Doe reviewed the contract and approved the invoice.",
        paragraph_index=4,
    )
    blocked_changed_claim = _add_claim(
        session,
        assertion=blocked_changed_assertion,
        text="Doe reviewed the contract and approved the invoice.",
        sequence_order=1,
    )
    segment = Segment(
        id=uuid.uuid4(),
        source_document=source_document,
        page_start=10,
        line_start=1,
        page_end=10,
        line_end=2,
        raw_text="A. Doe reviewed the contract.",
        normalized_text=normalize_for_match("A. Doe reviewed the contract."),
        speaker="A",
        segment_type="ANSWER_BLOCK",
    )
    link = SupportLink(
        claim_unit=blocked_changed_claim,
        segment=segment,
        sequence_order=1,
        link_type=LinkType.MANUAL,
        citation_text=None,
        user_confirmed=True,
    )
    session.add_all([segment, link])
    session.commit()

    return {
        "draft": draft,
        "legacy_unchanged": {
            "claim": legacy_unchanged_claim,
        },
        "blocked_changed": {
            "assertion": blocked_changed_assertion,
        },
    }


def _build_assertion(
    session: Session,
    *,
    draft: Draft,
    raw_text: str,
    paragraph_index: int,
) -> Assertion:
    assertion = Assertion(
        id=uuid.uuid4(),
        draft=draft,
        paragraph_index=paragraph_index,
        sentence_index=1,
        raw_text=raw_text,
        normalized_text=normalize_for_match(raw_text),
    )
    session.add(assertion)
    session.flush()
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
