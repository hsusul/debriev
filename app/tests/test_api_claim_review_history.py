from datetime import datetime, timedelta, timezone
import uuid

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.deps import get_db_session
from app.core.enums import ClaimReviewAction, ClaimType, DraftMode, SupportStatus
from app.main import create_app
from app.models import Assertion, Base, ClaimReviewDecision, ClaimUnit, Draft, Matter, VerificationRun
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


def test_claim_review_history_endpoint_surfaces_verification_and_decision_timeline(
    client: TestClient,
    session: Session,
) -> None:
    seeded = _seed_claim_history(session)

    response = client.get(f"/api/v1/claims/{seeded['claim_id']}/review-history")

    assert response.status_code == 200
    payload = response.json()
    assert payload["claim_id"] == str(seeded["claim_id"])
    assert payload["draft_id"] == str(seeded["draft_id"])
    assert payload["claim_text"] == "Doe delivered the notice."
    assert payload["assertion_context"] == "Doe delivered the notice."
    assert payload["support_status"] == SupportStatus.UNVERIFIED.value
    assert payload["review_disposition"] == "resolved"
    assert payload["latest_decision"]["action"] == ClaimReviewAction.MARK_FOR_REVISION.value
    assert len(payload["decision_history"]) == 2
    assert [decision["action"] for decision in payload["decision_history"]] == [
        ClaimReviewAction.MARK_FOR_REVISION.value,
        ClaimReviewAction.ACKNOWLEDGE_RISK.value,
    ]
    assert payload["latest_verification"]["verdict"] == SupportStatus.UNSUPPORTED.value
    assert payload["previous_verification"]["verdict"] == SupportStatus.AMBIGUOUS.value
    assert payload["latest_verification"]["reasoning_categories"] == ["scope_mismatch"]
    assert payload["previous_verification"]["reasoning_categories"] == ["weak_support"]
    assert payload["latest_verification"]["support_snapshot"]["provider_output"]["primary_anchor"] == "p.12:3-12:4"
    assert payload["previous_verification"]["support_snapshot"]["provider_output"]["primary_anchor"] == "p.10:1-10:2"
    assert payload["reasoning_categories"] == ["scope_mismatch"]
    assert payload["contradiction_flags"] == ["run_verdict_instability"]
    assert payload["claim_relationships"] == []
    assert payload["change_summary"] == {
        "latest_verdict": SupportStatus.UNSUPPORTED.value,
        "previous_verdict": SupportStatus.AMBIGUOUS.value,
        "verdict_changed": True,
        "latest_confidence_score": 0.2,
        "previous_confidence_score": 0.4,
        "confidence_changed": True,
        "latest_primary_anchor": "p.12:3-12:4",
        "previous_primary_anchor": "p.10:1-10:2",
        "primary_anchor_changed": True,
        "latest_flags": ["subject_mismatch"],
        "previous_flags": ["contextual_support_only"],
        "flags_changed": True,
        "latest_reasoning_categories": ["scope_mismatch"],
        "previous_reasoning_categories": ["weak_support"],
        "reasoning_categories_changed": True,
        "latest_support_assessment_count": 0,
        "previous_support_assessment_count": 0,
        "latest_excluded_link_count": 0,
        "previous_excluded_link_count": 0,
        "support_changed": True,
        "changed_since_last_run": True,
        "latest_decision_at": payload["latest_decision"]["created_at"],
        "latest_action": ClaimReviewAction.MARK_FOR_REVISION.value,
    }


def _seed_claim_history(session: Session) -> dict[str, uuid.UUID]:
    matter = Matter(name="Acme v. Doe", status="ACTIVE")
    draft = Draft(matter=matter, title="Review draft", mode=DraftMode.DRAFT)
    assertion = Assertion(
        draft=draft,
        paragraph_index=1,
        sentence_index=1,
        raw_text="Doe delivered the notice.",
        normalized_text=normalize_for_match("Doe delivered the notice."),
    )
    claim = ClaimUnit(
        assertion=assertion,
        text="Doe delivered the notice.",
        normalized_text=normalize_for_match("Doe delivered the notice."),
        claim_type=ClaimType.FACT,
        sequence_order=1,
        support_status=SupportStatus.UNVERIFIED,
    )
    session.add_all([matter, draft, assertion, claim])
    session.flush()

    base_time = datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc)
    first_run = VerificationRun(
        claim_unit=claim,
        model_version="test-model",
        prompt_version="test-prompt",
        deterministic_flags=["contextual_support_only"],
        reasoning_categories=["weak_support"],
        verdict=SupportStatus.AMBIGUOUS,
        reasoning="Only contextual overlap is linked.",
        support_snapshot_version=1,
        support_snapshot=_build_snapshot(
            claim_id=claim.id,
            draft_id=draft.id,
            matter_id=matter.id,
            primary_anchor="p.10:1-10:2",
        ),
        suggested_fix="Add direct testimony.",
        confidence_score=0.4,
        created_at=base_time,
    )
    second_run = VerificationRun(
        claim_unit=claim,
        model_version="test-model",
        prompt_version="test-prompt",
        deterministic_flags=["subject_mismatch"],
        reasoning_categories=["scope_mismatch"],
        verdict=SupportStatus.UNSUPPORTED,
        reasoning="Linked testimony is about a different actor.",
        support_snapshot_version=1,
        support_snapshot=_build_snapshot(
            claim_id=claim.id,
            draft_id=draft.id,
            matter_id=matter.id,
            primary_anchor="p.12:3-12:4",
        ),
        suggested_fix="Relink or revise the actor.",
        confidence_score=0.2,
        created_at=base_time + timedelta(minutes=10),
    )
    first_decision = ClaimReviewDecision(
        claim_unit=claim,
        draft=draft,
        verification_run=first_run,
        action=ClaimReviewAction.ACKNOWLEDGE_RISK,
        note=None,
        proposed_replacement_text=None,
        created_at=base_time + timedelta(minutes=12),
    )
    second_decision = ClaimReviewDecision(
        claim_unit=claim,
        draft=draft,
        verification_run=second_run,
        action=ClaimReviewAction.MARK_FOR_REVISION,
        note="Split the actor mismatch before filing.",
        proposed_replacement_text=None,
        created_at=base_time + timedelta(minutes=15),
    )

    session.add_all([first_run, second_run, first_decision, second_decision])
    session.commit()

    return {"draft_id": draft.id, "claim_id": claim.id}


def _build_snapshot(
    *,
    claim_id,
    draft_id,
    matter_id,
    primary_anchor: str,
) -> dict[str, object]:
    return {
        "claim_scope": {
            "claim_id": str(claim_id),
            "draft_id": str(draft_id),
            "matter_id": str(matter_id),
            "evidence_bundle_id": None,
            "scope_kind": "matter_fallback",
            "allowed_source_document_ids": [],
        },
        "valid_support_links": [],
        "excluded_support_links": [],
        "support_items": [],
        "citations": [],
        "provider_output": {
            "primary_anchor": primary_anchor,
            "support_assessments": [],
        },
    }
