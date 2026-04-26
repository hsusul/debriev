import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.deps import get_db_session
from app.core.enums import ClaimType, DraftMode, SupportStatus
from app.main import create_app
from app.models import Assertion, Base, ClaimUnit, Draft, Matter, VerificationRun
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


def test_claim_review_decision_endpoint_persists_mark_for_revision_and_returns_queue_state(
    client: TestClient,
    session: Session,
) -> None:
    seeded = _seed_reviewable_claims(session)

    response = client.post(
        f"/api/v1/claims/{seeded['first_claim_id']}/decisions",
        json={
            "action": "mark_for_revision",
            "note": "Split the actor/date mismatch before final review.",
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["decision"]["claim_unit_id"] == str(seeded["first_claim_id"])
    assert payload["decision"]["draft_id"] == str(seeded["draft_id"])
    assert payload["decision"]["verification_run_id"] == str(seeded["first_run_id"])
    assert payload["decision"]["action"] == "mark_for_revision"
    assert payload["decision"]["note"] == "Split the actor/date mismatch before final review."
    assert payload["decision"]["proposed_replacement_text"] is None
    assert payload["claim_review_state"] == {
        "claim_id": str(seeded["first_claim_id"]),
        "draft_id": str(seeded["draft_id"]),
        "review_status": "reviewed",
        "latest_action": "mark_for_revision",
        "decision_count": 1,
        "latest_verification_run_id": str(seeded["first_run_id"]),
        "latest_verdict": SupportStatus.UNSUPPORTED.value,
        "removed_from_active_queue": True,
    }
    assert payload["draft_queue"] == {
        "draft_id": str(seeded["draft_id"]),
        "total_flagged_claims": 2,
        "resolved_flagged_claims": 1,
        "remaining_flagged_claims": 1,
        "next_claim_id": str(seeded["second_claim_id"]),
    }


def test_claim_review_decision_endpoint_rejects_invalid_payload(
    client: TestClient,
    session: Session,
) -> None:
    seeded = _seed_reviewable_claims(session)

    response = client.post(
        f"/api/v1/claims/{seeded['first_claim_id']}/decisions",
        json={"action": "resolve_with_edit"},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Resolve with edit requires proposed replacement text."}


def _seed_reviewable_claims(session: Session) -> dict[str, uuid.UUID]:
    matter = Matter(name="Acme v. Doe", status="ACTIVE")
    draft = Draft(matter=matter, title="Review draft", mode=DraftMode.DRAFT)

    first_assertion = Assertion(
        draft=draft,
        paragraph_index=1,
        sentence_index=1,
        raw_text="Doe delivered the notice.",
        normalized_text=normalize_for_match("Doe delivered the notice."),
    )
    first_claim = ClaimUnit(
        assertion=first_assertion,
        text="Doe delivered the notice.",
        normalized_text=normalize_for_match("Doe delivered the notice."),
        claim_type=ClaimType.FACT,
        sequence_order=1,
        support_status=SupportStatus.UNVERIFIED,
    )
    first_run = VerificationRun(
        claim_unit=first_claim,
        model_version="test-model",
        prompt_version="test-prompt",
        deterministic_flags=["subject_mismatch"],
        verdict=SupportStatus.UNSUPPORTED,
        reasoning="Different actor in the linked testimony.",
        support_snapshot_version=None,
        support_snapshot=None,
        suggested_fix="Relink or revise the actor.",
        confidence_score=0.2,
    )

    second_assertion = Assertion(
        draft=draft,
        paragraph_index=2,
        sentence_index=1,
        raw_text="Doe signed the contract.",
        normalized_text=normalize_for_match("Doe signed the contract."),
    )
    second_claim = ClaimUnit(
        assertion=second_assertion,
        text="Doe signed the contract.",
        normalized_text=normalize_for_match("Doe signed the contract."),
        claim_type=ClaimType.FACT,
        sequence_order=1,
        support_status=SupportStatus.UNVERIFIED,
    )
    VerificationRun(
        claim_unit=second_claim,
        model_version="test-model",
        prompt_version="test-prompt",
        deterministic_flags=["contextual_support_only"],
        verdict=SupportStatus.AMBIGUOUS,
        reasoning="Only contextual overlap is linked.",
        support_snapshot_version=None,
        support_snapshot=None,
        suggested_fix="Add direct testimony.",
        confidence_score=0.4,
    )

    session.add_all([matter, draft, first_assertion, second_assertion])
    session.commit()

    return {
        "draft_id": draft.id,
        "first_claim_id": first_claim.id,
        "first_run_id": first_run.id,
        "second_claim_id": second_claim.id,
    }
