import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.enums import ClaimType, DraftMode, LinkType, ParserStatus, ReExtractionRunKind, SourceType, SupportStatus
from app.models import (
    Assertion,
    Base,
    CURRENT_REEXTRACTION_RUN_SNAPSHOT_VERSION,
    ClaimUnit,
    Draft,
    Matter,
    Segment,
    SourceDocument,
    SupportLink,
)
from app.repositories.reextraction_runs import ReExtractionRunRepository
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


def test_preview_draft_surfaces_mixed_ready_unchanged_and_blocked_assertions(session: Session) -> None:
    workspace = _seed_batch_reextract_draft(session)

    result = ClaimReExtractionService(session).preview_draft(workspace["draft"].id, mode="structured")

    assert result.draft_id == workspace["draft"].id
    assert result.run_id is not None
    assert result.requested_mode == "structured"
    assert result.total_assertions == 4
    assert result.ready_assertions == 2
    assert result.unchanged_assertions == 1
    assert result.blocked_assertions == 1
    assert result.materially_changed_assertions == 2
    assert result.legacy_unversioned_assertions == 3
    assert [item.status for item in result.items] == [
        "ready",
        "ready",
        "unchanged",
        "blocked",
    ]

    ready_replace = result.items[0]
    assert ready_replace.assertion_id == workspace["legacy_changed"]["assertion"].id
    assert ready_replace.materially_changed is True
    assert ready_replace.apply_requires_replacement is True
    assert ready_replace.can_apply is True
    assert ready_replace.existing_claim_count == 1
    assert ready_replace.proposed_claim_count == 2

    ready_metadata_only = result.items[1]
    assert ready_metadata_only.assertion_id == workspace["legacy_unchanged"]["assertion"].id
    assert ready_metadata_only.materially_changed is False
    assert ready_metadata_only.apply_requires_replacement is False
    assert ready_metadata_only.can_apply is True
    assert ready_metadata_only.existing_metadata.status == "legacy_unversioned"

    skipped_current = result.items[2]
    assert skipped_current.assertion_id == workspace["versioned_current"]["assertion"].id
    assert skipped_current.status == "unchanged"
    assert skipped_current.existing_metadata.status == "versioned"
    assert skipped_current.existing_metadata.strategy == "structured"
    assert skipped_current.existing_metadata.version == CURRENT_EXTRACTION_VERSION

    blocked = result.items[3]
    assert blocked.assertion_id == workspace["blocked_changed"]["assertion"].id
    assert blocked.status == "blocked"
    assert blocked.materially_changed is True
    assert blocked.apply_requires_replacement is True
    assert blocked.can_apply is False
    assert blocked.blocked_reasons == [
        "Re-extraction cannot replace persisted claim units while support links are attached."
    ]

    persisted_run = ReExtractionRunRepository(session).list_by_draft(workspace["draft"].id)[0]
    assert persisted_run.id == result.run_id
    assert persisted_run.run_kind == ReExtractionRunKind.PREVIEW
    assert persisted_run.requested_mode == "structured"
    assert persisted_run.extraction_version == CURRENT_EXTRACTION_VERSION
    assert persisted_run.snapshot_version == CURRENT_REEXTRACTION_RUN_SNAPSHOT_VERSION
    assert persisted_run.total_assertions == 4
    assert persisted_run.ready_assertions == 2
    assert persisted_run.unchanged_assertions == 1
    assert persisted_run.blocked_assertions == 1
    assert persisted_run.materially_changed_assertions == 2
    assert persisted_run.legacy_unversioned_assertions == 3
    assert persisted_run.snapshot["run_kind"] == "PREVIEW"
    assert persisted_run.snapshot["summary"] == {
        "total_assertions": 4,
        "ready_assertions": 2,
        "unchanged_assertions": 1,
        "blocked_assertions": 1,
        "materially_changed_assertions": 2,
        "legacy_unversioned_assertions": 3,
        "applied_assertions": 0,
        "skipped_assertions": 0,
        "replaced_assertions": 0,
        "metadata_only_assertions": 0,
    }
    assert [item["status"] for item in persisted_run.snapshot["items"]] == [
        "ready",
        "ready",
        "unchanged",
        "blocked",
    ]


def test_apply_draft_only_applies_safe_assertions_and_reports_blocked_ones(session: Session) -> None:
    workspace = _seed_batch_reextract_draft(session)

    result = ClaimReExtractionService(session).apply_draft(workspace["draft"].id, mode="structured")

    assert result.draft_id == workspace["draft"].id
    assert result.run_id is not None
    assert result.requested_mode == "structured"
    assert result.total_assertions == 4
    assert result.applied_assertions == 2
    assert result.skipped_assertions == 1
    assert result.blocked_assertions == 1
    assert result.replaced_assertions == 1
    assert result.metadata_only_assertions == 1
    assert [item.status for item in result.items] == [
        "applied",
        "applied",
        "skipped",
        "blocked",
    ]

    replaced_item = result.items[0]
    assert replaced_item.assertion_id == workspace["legacy_changed"]["assertion"].id
    assert replaced_item.claims_replaced is True
    assert replaced_item.metadata_updated is True
    assert [claim.text for claim in replaced_item.resulting_claims] == [
        "Doe signed the contract",
        "Doe approved the invoice.",
    ]

    metadata_only_item = result.items[1]
    assert metadata_only_item.assertion_id == workspace["legacy_unchanged"]["assertion"].id
    assert metadata_only_item.claims_replaced is False
    assert metadata_only_item.metadata_updated is True
    assert [claim.claim_id for claim in metadata_only_item.resulting_claims] == [
        workspace["legacy_unchanged"]["claim"].id
    ]

    skipped_item = result.items[2]
    assert skipped_item.assertion_id == workspace["versioned_current"]["assertion"].id
    assert skipped_item.claims_replaced is False
    assert skipped_item.metadata_updated is False
    assert skipped_item.notes == [
        "Assertion already matches the selected extraction strategy/version and needs no migration."
    ]

    blocked_item = result.items[3]
    assert blocked_item.assertion_id == workspace["blocked_changed"]["assertion"].id
    assert blocked_item.status == "blocked"
    assert blocked_item.blocked_reasons == [
        "Re-extraction cannot replace persisted claim units while support links are attached."
    ]
    assert blocked_item.metadata_updated is False

    session.refresh(workspace["legacy_changed"]["assertion"])
    assert workspace["legacy_changed"]["assertion"].extraction_strategy == "structured"
    assert workspace["legacy_changed"]["assertion"].extraction_version == CURRENT_EXTRACTION_VERSION
    assert workspace["legacy_changed"]["assertion"].extraction_snapshot is not None
    assert [claim.text for claim in workspace["legacy_changed"]["assertion"].claim_units] == [
        "Doe signed the contract",
        "Doe approved the invoice.",
    ]

    session.refresh(workspace["legacy_unchanged"]["assertion"])
    assert workspace["legacy_unchanged"]["assertion"].extraction_strategy == "structured"
    assert workspace["legacy_unchanged"]["assertion"].extraction_version == CURRENT_EXTRACTION_VERSION
    assert workspace["legacy_unchanged"]["assertion"].extraction_snapshot is not None
    assert workspace["legacy_unchanged"]["assertion"].extraction_snapshot["claims"][0]["claim_id"] == str(
        workspace["legacy_unchanged"]["claim"].id
    )

    session.refresh(workspace["versioned_current"]["assertion"])
    assert workspace["versioned_current"]["assertion"].extraction_strategy == "structured"
    assert workspace["versioned_current"]["assertion"].extraction_version == CURRENT_EXTRACTION_VERSION
    assert workspace["versioned_current"]["assertion"].extraction_snapshot is not None

    session.refresh(workspace["blocked_changed"]["assertion"])
    assert workspace["blocked_changed"]["assertion"].extraction_strategy is None
    assert workspace["blocked_changed"]["assertion"].extraction_version is None
    assert workspace["blocked_changed"]["assertion"].extraction_snapshot is None

    persisted_run = ReExtractionRunRepository(session).list_by_draft(workspace["draft"].id)[0]
    assert persisted_run.id == result.run_id
    assert persisted_run.run_kind == ReExtractionRunKind.APPLY
    assert persisted_run.requested_mode == "structured"
    assert persisted_run.extraction_version == CURRENT_EXTRACTION_VERSION
    assert persisted_run.snapshot_version == CURRENT_REEXTRACTION_RUN_SNAPSHOT_VERSION
    assert persisted_run.total_assertions == 4
    assert persisted_run.applied_assertions == 2
    assert persisted_run.skipped_assertions == 1
    assert persisted_run.blocked_assertions == 1
    assert persisted_run.replaced_assertions == 1
    assert persisted_run.metadata_only_assertions == 1
    assert persisted_run.snapshot["run_kind"] == "APPLY"
    assert persisted_run.snapshot["summary"] == {
        "total_assertions": 4,
        "ready_assertions": 0,
        "unchanged_assertions": 0,
        "blocked_assertions": 1,
        "materially_changed_assertions": 2,
        "legacy_unversioned_assertions": 3,
        "applied_assertions": 2,
        "skipped_assertions": 1,
        "replaced_assertions": 1,
        "metadata_only_assertions": 1,
    }
    assert [item["status"] for item in persisted_run.snapshot["items"]] == [
        "applied",
        "applied",
        "skipped",
        "blocked",
    ]


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
    legacy_changed_claim = _add_claim(
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
        "legacy_changed": {
            "assertion": legacy_changed_assertion,
            "claim": legacy_changed_claim,
        },
        "legacy_unchanged": {
            "assertion": legacy_unchanged_assertion,
            "claim": legacy_unchanged_claim,
        },
        "versioned_current": {
            "assertion": versioned_current_assertion,
        },
        "blocked_changed": {
            "assertion": blocked_changed_assertion,
            "claim": blocked_changed_claim,
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
