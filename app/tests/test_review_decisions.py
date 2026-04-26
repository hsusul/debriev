import uuid

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.enums import ClaimReviewAction, ClaimType, DraftMode, SupportStatus
from app.core.exceptions import ValidationError
from app.models import Assertion, Base, ClaimReviewDecision, ClaimUnit, Draft, Matter, VerificationRun
from app.schemas.review_decision import ClaimReviewDecisionCreate
from app.services.parsing.normalization import normalize_for_match
from app.services.workflows.review_decisions import ClaimReviewDecisionService


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


def test_acknowledge_risk_persists_decision_and_updates_queue_counts(session: Session) -> None:
    seeded = _seed_reviewable_claims(session)

    result = ClaimReviewDecisionService(session).record_decision(
        seeded["first_claim_id"],
        ClaimReviewDecisionCreate(action=ClaimReviewAction.ACKNOWLEDGE_RISK),
    )
    session.commit()

    stored = session.scalar(
        select(ClaimReviewDecision).where(ClaimReviewDecision.id == result.decision.id)
    )

    assert stored is not None
    assert stored.claim_unit_id == seeded["first_claim_id"]
    assert stored.draft_id == seeded["draft_id"]
    assert stored.verification_run_id == seeded["first_run_id"]
    assert stored.action == ClaimReviewAction.ACKNOWLEDGE_RISK
    assert stored.note is None
    assert stored.proposed_replacement_text is None
    assert result.claim_review_state.latest_verdict == SupportStatus.UNSUPPORTED
    assert result.claim_review_state.removed_from_active_queue is True
    assert result.draft_queue.total_flagged_claims == 2
    assert result.draft_queue.resolved_flagged_claims == 1
    assert result.draft_queue.remaining_flagged_claims == 1
    assert result.draft_queue.next_claim_id == seeded["second_claim_id"]


def test_mark_for_revision_persists_note(session: Session) -> None:
    seeded = _seed_reviewable_claims(session)

    result = ClaimReviewDecisionService(session).record_decision(
        seeded["first_claim_id"],
        ClaimReviewDecisionCreate(
            action=ClaimReviewAction.MARK_FOR_REVISION,
            note="Split the actor/date mismatch before final review.",
        ),
    )
    session.commit()

    assert result.decision.action == ClaimReviewAction.MARK_FOR_REVISION
    assert result.decision.note == "Split the actor/date mismatch before final review."
    assert result.decision.proposed_replacement_text is None


def test_resolve_with_edit_persists_proposed_text(session: Session) -> None:
    seeded = _seed_reviewable_claims(session)

    result = ClaimReviewDecisionService(session).record_decision(
        seeded["first_claim_id"],
        ClaimReviewDecisionCreate(
            action=ClaimReviewAction.RESOLVE_WITH_EDIT,
            proposed_replacement_text="Doe delivered the notice on March 3.",
            note="Narrowed to the supported date.",
        ),
    )
    session.commit()

    assert result.decision.action == ClaimReviewAction.RESOLVE_WITH_EDIT
    assert result.decision.proposed_replacement_text == "Doe delivered the notice on March 3."
    assert result.decision.note == "Narrowed to the supported date."


def test_invalid_action_payloads_are_rejected_cleanly(session: Session) -> None:
    seeded = _seed_reviewable_claims(session)
    service = ClaimReviewDecisionService(session)

    with pytest.raises(ValidationError, match="Mark for revision requires a note"):
        service.record_decision(
            seeded["first_claim_id"],
            ClaimReviewDecisionCreate(action=ClaimReviewAction.MARK_FOR_REVISION),
        )

    with pytest.raises(ValidationError, match="Resolve with edit requires proposed replacement text"):
        service.record_decision(
            seeded["first_claim_id"],
            ClaimReviewDecisionCreate(action=ClaimReviewAction.RESOLVE_WITH_EDIT),
        )


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

    third_assertion = Assertion(
        draft=draft,
        paragraph_index=3,
        sentence_index=1,
        raw_text="Smith reviewed the contract.",
        normalized_text=normalize_for_match("Smith reviewed the contract."),
    )
    third_claim = ClaimUnit(
        assertion=third_assertion,
        text="Smith reviewed the contract.",
        normalized_text=normalize_for_match("Smith reviewed the contract."),
        claim_type=ClaimType.FACT,
        sequence_order=1,
        support_status=SupportStatus.UNVERIFIED,
    )
    VerificationRun(
        claim_unit=third_claim,
        model_version="test-model",
        prompt_version="test-prompt",
        deterministic_flags=[],
        verdict=SupportStatus.SUPPORTED,
        reasoning="Direct testimony supports the claim.",
        support_snapshot_version=None,
        support_snapshot=None,
        suggested_fix=None,
        confidence_score=0.8,
    )

    session.add_all([matter, draft, first_assertion, second_assertion, third_assertion])
    session.commit()

    return {
        "draft_id": draft.id,
        "first_claim_id": first_claim.id,
        "first_run_id": first_run.id,
        "second_claim_id": second_claim.id,
    }
