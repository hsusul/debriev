import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.deps import get_db_session
from app.config import get_settings
from app.core.enums import ClaimType, DraftMode, LinkType, ParserStatus, SourceType, SupportStatus
from app.main import create_app
from app.models import Assertion, Base, ClaimUnit, Draft, Matter, Segment, SourceDocument, SupportLink
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


def test_verify_endpoint_returns_structured_fields_with_provider_backing(
    client: TestClient,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim_id = _seed_claim_with_support(
        session,
        claim_text="Smith signed the contract.",
        support_rows=[
            {
                "raw_text": "Q. Did you review the contract?",
                "page_start": 10,
                "line_start": 1,
                "page_end": 10,
                "line_end": 2,
                "speaker": "Q",
                "segment_type": "QUESTION_BLOCK",
                "sequence_order": 1,
            },
            {
                "raw_text": "A. Smith signed the contract on March 1.",
                "page_start": 10,
                "line_start": 3,
                "page_end": 10,
                "line_end": 4,
                "speaker": "A",
                "segment_type": "ANSWER_BLOCK",
                "sequence_order": 2,
            },
        ],
    )
    _configure_provider(monkeypatch, provider="openai", api_key="test-key")

    response = client.post(f"/api/v1/claims/{claim_id}/verify", json={})

    assert response.status_code == 201
    payload = response.json()
    assert payload["primary_anchor"] == "p.10:3-10:4"
    assert [assessment["anchor"] for assessment in payload["support_assessments"]] == [
        "p.10:1-10:2",
        "p.10:3-10:4",
    ]
    assert payload["support_assessments"][0]["role"] == "contextual"
    assert payload["support_assessments"][1]["role"] == "primary"


def test_verify_endpoint_deterministic_only_returns_empty_structured_fields(
    client: TestClient,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim_id = _seed_claim_with_support(
        session,
        claim_text="Smith signed the contract.",
        support_rows=[
            {
                "raw_text": "A. Smith signed the contract on March 1.",
                "page_start": 20,
                "line_start": 3,
                "page_end": 20,
                "line_end": 4,
                "speaker": "A",
                "segment_type": "ANSWER_BLOCK",
                "sequence_order": 1,
            }
        ],
    )
    _configure_provider(monkeypatch, provider=None, api_key=None)

    response = client.post(f"/api/v1/claims/{claim_id}/verify", json={})

    assert response.status_code == 201
    payload = response.json()
    assert payload["primary_anchor"] is None
    assert payload["support_assessments"] == []


def test_verification_run_history_endpoint_surfaces_persisted_support_snapshot(
    client: TestClient,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim_id = _seed_claim_with_support(
        session,
        claim_text="Smith signed the contract.",
        support_rows=[
            {
                "raw_text": "A. Smith signed the contract on March 1.",
                "page_start": 30,
                "line_start": 3,
                "page_end": 30,
                "line_end": 4,
                "speaker": "A",
                "segment_type": "ANSWER_BLOCK",
                "sequence_order": 1,
            }
        ],
    )
    _configure_provider(monkeypatch, provider="openai", api_key="test-key")

    verify_response = client.post(f"/api/v1/claims/{claim_id}/verify", json={})
    history_response = client.get(f"/api/v1/claims/{claim_id}/verification-runs")

    assert verify_response.status_code == 201
    assert history_response.status_code == 200
    runs = history_response.json()
    assert len(runs) == 1
    assert "primary_anchor" not in runs[0]
    assert "support_assessments" not in runs[0]
    assert runs[0]["support_snapshot_status"] == "versioned_v1"
    assert runs[0]["support_snapshot_note"] is None
    assert runs[0]["support_snapshot_version"] == 1
    assert runs[0]["support_snapshot"]["provider_output"]["primary_anchor"] == "p.30:3-30:4"
    assert runs[0]["support_snapshot"]["support_items"][0]["anchor"] == "p.30:3-30:4"
    assert runs[0]["support_snapshot"]["excluded_support_links"] == []
    assert runs[0]["claim_unit_id"] == str(claim_id)


def _configure_provider(
    monkeypatch: pytest.MonkeyPatch,
    *,
    provider: str | None,
    api_key: str | None,
) -> None:
    if provider is None:
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
    else:
        monkeypatch.setenv("LLM_PROVIDER", provider)

    if api_key is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", api_key)

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()


def _seed_claim_with_support(
    session: Session,
    *,
    claim_text: str,
    support_rows: list[dict[str, object]],
) -> uuid.UUID:
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
        raw_text=claim_text,
        normalized_text=normalize_for_match(claim_text),
    )
    claim = ClaimUnit(
        assertion=assertion,
        text=claim_text,
        normalized_text=normalize_for_match(claim_text),
        claim_type=ClaimType.FACT,
        sequence_order=1,
        support_status=SupportStatus.UNVERIFIED,
    )
    session.add_all([matter, source_document, draft, assertion, claim])
    session.flush()

    for row in support_rows:
        segment = Segment(
            id=uuid.uuid4(),
            source_document_id=source_document.id,
            page_start=row["page_start"],
            line_start=row["line_start"],
            page_end=row["page_end"],
            line_end=row["line_end"],
            raw_text=row["raw_text"],
            normalized_text=normalize_for_match(str(row["raw_text"])),
            speaker=row["speaker"],
            segment_type=row["segment_type"],
        )
        session.add(segment)
        session.flush()
        session.add(
            SupportLink(
                id=uuid.uuid4(),
                claim_unit_id=claim.id,
                segment_id=segment.id,
                sequence_order=row["sequence_order"],
                link_type=LinkType.MANUAL,
                citation_text=None,
                user_confirmed=True,
            )
        )

    session.commit()
    return claim.id
