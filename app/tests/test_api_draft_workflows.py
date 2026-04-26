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
from app.models import (
    Assertion,
    Base,
    ClaimReviewDecision,
    ClaimUnit,
    Draft,
    EvidenceBundle,
    Matter,
    Segment,
    SourceDocument,
    SupportLink,
)
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


def test_draft_endpoints_round_trip_evidence_bundle_reference(
    client: TestClient,
    session: Session,
) -> None:
    matter = Matter(name="Acme v. Doe", status="ACTIVE")
    source_document = SourceDocument(
        matter=matter,
        file_name="declaration.txt",
        source_type=SourceType.DECLARATION,
        raw_file_path="/tmp/declaration.txt",
        parser_status=ParserStatus.COMPLETED,
        parser_confidence=0.85,
    )
    bundle = EvidenceBundle(
        matter=matter,
        name="Review Record",
        description="Initial declaration scope.",
    )
    bundle.source_documents.append(source_document)
    session.add_all([matter, source_document, bundle])
    session.commit()

    create_response = client.post(
        f"/api/v1/matters/{matter.id}/drafts",
        json={
            "title": "Scoped draft",
            "mode": DraftMode.DRAFT.value,
            "evidence_bundle_id": str(bundle.id),
        },
    )

    assert create_response.status_code == 201
    created_payload = create_response.json()
    assert created_payload["matter_id"] == str(matter.id)
    assert created_payload["evidence_bundle_id"] == str(bundle.id)

    get_response = client.get(f"/api/v1/drafts/{created_payload['id']}")

    assert get_response.status_code == 200
    fetched_payload = get_response.json()
    assert fetched_payload["id"] == created_payload["id"]
    assert fetched_payload["evidence_bundle_id"] == str(bundle.id)


def test_create_draft_from_text_creates_reviewable_draft(
    client: TestClient,
    session: Session,
) -> None:
    response = client.post(
        "/api/v1/drafts",
        json={
            "draft_text": "Breach of Contract\n\nDoe signed the agreement on March 1.\n\nDoe never delivered the notice.",
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["draft_id"] is not None
    assert payload["matter_id"] is not None
    assert payload["title"] == "Breach of Contract"
    assert payload["assertion_count"] == 3
    assert payload["claim_count"] >= 2

    draft = session.get(Draft, uuid.UUID(payload["draft_id"]))
    assert draft is not None
    assert draft.title == "Breach of Contract"
    assert draft.matter_id == uuid.UUID(payload["matter_id"])

    assertions = session.query(Assertion).filter(Assertion.draft_id == draft.id).order_by(Assertion.paragraph_index).all()
    assert [assertion.raw_text for assertion in assertions] == [
        "Breach of Contract",
        "Doe signed the agreement on March 1.",
        "Doe never delivered the notice.",
    ]

    claims = session.query(ClaimUnit).join(Assertion).filter(Assertion.draft_id == draft.id).all()
    assert len(claims) == payload["claim_count"]
    assert all(claim.support_status == SupportStatus.UNVERIFIED for claim in claims)


def test_citation_verification_endpoint_surfaces_recognized_but_unresolved_partial_citation(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_provider(monkeypatch, provider=None, api_key=None)

    response = client.post(
        "/api/v1/citation-verification",
        json={
            "draft_text": (
                "Citation Memo\n\n"
                "Under Smith v. Jones, Doe had to provide notice.\n\n"
                "Doe always had to provide notice."
            ),
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["draft_id"] is not None
    assert payload["matter_id"] is not None
    assert payload["title"] == "Citation Memo"
    assert payload["review_run_id"] is not None
    assert payload["reviewed_at"] is not None
    assert payload["summary"]["total_claims"] >= 2
    assert payload["summary"]["total_cited_propositions"] == 1
    assert payload["summary"]["flagged_citation_count"] == 1
    assert payload["summary"]["verdict_counts"]["ambiguous"] == 1
    assert payload["summary"]["authority_status_counts"]["authority_unverified"] == 1
    assert payload["summary"]["authority_status_counts"]["citation_recognized"] == 1
    assert payload["summary"]["authority_status_counts"]["authority_candidate_parsed"] == 0
    assert payload["summary"]["authority_status_counts"]["authority_matched"] == 0
    assert [item["citation_text"] for item in payload["citations"]] == ["Smith v. Jones"]
    assert payload["citations"][0]["proposition_text"] == "Under Smith v. Jones, Doe had to provide notice."
    assert payload["citations"][0]["authority_status"] == "citation_recognized"
    assert payload["citations"][0]["authority_match_status"] == "recognized_only"
    assert payload["citations"][0]["parsed_authority"] == {
        "case_name": "Smith v. Jones",
        "reporter_volume": None,
        "reporter_abbreviation": None,
        "first_page": None,
        "court": None,
        "year": None,
    }
    assert payload["citations"][0]["normalized_authority_reference"] == "smith v jones"
    assert payload["citations"][0]["matched_authority"] is None
    assert payload["citations"][0]["proposition_verdict"] == SupportStatus.AMBIGUOUS.value
    assert payload["citations"][0]["reasoning"] is not None
    assert payload["citations"][0]["support_snippet"] is None


def test_citation_verification_endpoint_matches_known_authority_from_mvp_catalog(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_provider(monkeypatch, provider=None, api_key=None)

    response = client.post(
        "/api/v1/citation-verification",
        json={
            "draft_text": (
                "Brown v. Board of Education, 347 U.S. 483 (1954), held that segregation in public schools "
                "violates equal protection."
            ),
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["summary"]["total_cited_propositions"] == 1
    assert payload["summary"]["flagged_citation_count"] == 1
    assert payload["summary"]["verdict_counts"]["ambiguous"] == 1
    assert payload["summary"]["authority_status_counts"]["authority_unverified"] == 0
    assert payload["summary"]["authority_status_counts"]["authority_matched"] == 1
    assert payload["citations"][0]["citation_text"] == "Brown v. Board of Education, 347 U.S. 483 (1954)"
    assert payload["citations"][0]["authority_status"] == "authority_matched"
    assert payload["citations"][0]["authority_match_status"] == "matched"
    assert payload["citations"][0]["parsed_authority"] == {
        "case_name": "Brown v. Board of Education",
        "reporter_volume": "347",
        "reporter_abbreviation": "U.S.",
        "first_page": "483",
        "court": None,
        "year": 1954,
    }
    assert (
        payload["citations"][0]["normalized_authority_reference"]
        == "brown v board of education|347|u.s.|483|1954"
    )
    assert payload["citations"][0]["matched_authority"] == {
        "authority_id": "brown-v-board-of-education-347-us-483",
        "canonical_name": "Brown v. Board of Education",
        "canonical_citation": "Brown v. Board of Education, 347 U.S. 483 (1954)",
        "reporter_volume": "347",
        "reporter_abbreviation": "U.S.",
        "first_page": "483",
        "court": None,
        "year": 1954,
        "source_name": "debriev_mvp_authority_catalog",
    }
    assert payload["citations"][0]["proposition_verdict"] == SupportStatus.AMBIGUOUS.value


def test_citation_verification_endpoint_surfaces_parseable_but_unmatched_authority(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_provider(monkeypatch, provider=None, api_key=None)

    response = client.post(
        "/api/v1/citation-verification",
        json={
            "draft_text": "Brown v. Davis, 999 U.S. 1 (2001), held that negligence is always enough.",
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["summary"]["total_cited_propositions"] == 1
    assert payload["summary"]["authority_status_counts"]["authority_unverified"] == 1
    assert payload["summary"]["authority_status_counts"]["authority_candidate_parsed"] == 1
    assert payload["summary"]["authority_status_counts"]["citation_recognized"] == 0
    assert payload["summary"]["authority_status_counts"]["authority_matched"] == 0
    assert payload["citations"][0]["authority_status"] == "authority_candidate_parsed"
    assert payload["citations"][0]["authority_match_status"] == "no_match"
    assert payload["citations"][0]["parsed_authority"] == {
        "case_name": "Brown v. Davis",
        "reporter_volume": "999",
        "reporter_abbreviation": "U.S.",
        "first_page": "1",
        "court": None,
        "year": 2001,
    }
    assert payload["citations"][0]["matched_authority"] is None


def test_citation_verification_endpoint_pairs_multiple_citations_and_summary_counts_align(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_provider(monkeypatch, provider=None, api_key=None)

    response = client.post(
        "/api/v1/citation-verification",
        json={
            "draft_text": (
                "Brown v. Board of Education, 347 U.S. 483 (1954), held that segregation in public schools "
                "violates equal protection. Brown v. Davis, 999 U.S. 1 (2001), held that negligence is always enough."
            ),
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["summary"]["total_cited_propositions"] == 2
    assert payload["summary"]["flagged_citation_count"] == 2
    assert [item["citation_text"] for item in payload["citations"]] == [
        "Brown v. Board of Education, 347 U.S. 483 (1954)",
        "Brown v. Davis, 999 U.S. 1 (2001)",
    ]
    assert [item["proposition_text"] for item in payload["citations"]] == [
        (
            "Brown v. Board of Education, 347 U.S. 483 (1954), held that segregation in public schools "
            "violates equal protection."
        ),
        "Brown v. Davis, 999 U.S. 1 (2001), held that negligence is always enough.",
    ]
    assert [item["authority_status"] for item in payload["citations"]] == [
        "authority_matched",
        "authority_candidate_parsed",
    ]
    assert payload["summary"]["authority_status_counts"] == {
        "authority_unverified": 1,
        "citation_recognized": 0,
        "authority_candidate_parsed": 1,
        "authority_matched": 1,
        "linked_authority_support_present": 0,
        "not_reviewed": 0,
    }
    verdict_total = sum(payload["summary"]["verdict_counts"].values())
    authority_total = sum(payload["summary"]["authority_status_counts"].values()) - payload["summary"][
        "authority_status_counts"
    ]["authority_unverified"]
    assert verdict_total == len(payload["citations"])
    assert authority_total == len(payload["citations"])


def test_compile_endpoint_returns_structured_compile_output(
    client: TestClient,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_provider(monkeypatch, provider="openai", api_key="test-key")
    draft_id = _seed_reviewable_draft(session)

    response = client.post(f"/api/v1/drafts/{draft_id}/compile")

    assert response.status_code == 200
    payload = response.json()
    assert payload["draft_id"] == str(draft_id)
    assert payload["total_claims"] == 5
    assert payload["verdict_counts"] == {
        "supported": 1,
        "partially_supported": 1,
        "overstated": 0,
        "ambiguous": 1,
        "unsupported": 2,
        "unverified": 0,
    }
    assert [claim["claim_text"] for claim in payload["flagged_claims"]] == [
        "Doe signed the contract on March 1.",
        "Doe delivered the notice.",
        "Doe admitted the error.",
    ]
    ambiguous_claim = next(
        claim for claim in payload["flagged_claims"] if claim["claim_text"] == "Doe signed the contract on March 1."
    )
    assert ambiguous_claim["verdict"] == SupportStatus.AMBIGUOUS.value
    assert ambiguous_claim["primary_anchor"] == "p.30:1-30:2"
    assert [assessment["anchor"] for assessment in ambiguous_claim["support_assessments"]] == [
        "p.30:1-30:2",
        "p.30:3-30:4",
    ]
    assert [assessment["role"] for assessment in ambiguous_claim["support_assessments"]] == [
        "contextual",
        "contextual",
    ]


def test_review_endpoint_returns_structured_review_output(
    client: TestClient,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_provider(monkeypatch, provider=None, api_key=None)
    draft_id = _seed_reviewable_draft(session)

    response = client.post(f"/api/v1/drafts/{draft_id}/review")

    assert response.status_code == 200
    payload = response.json()
    assert payload["draft_id"] == str(draft_id)
    assert payload["total_claims"] == 5
    assert payload["freshness"]["state_source"] == "fresh_execution"
    assert payload["freshness"]["has_persisted_review_runs"] is True
    assert payload["freshness"]["last_review_run_at"] is not None
    assert payload["freshness"]["latest_review_run_id"] is not None
    assert payload["freshness"]["latest_review_run_status"] == "COMPLETED"
    assert payload["freshness"]["latest_decision_at"] is None
    assert payload["freshness"]["has_decisions_after_latest_run"] is False
    assert payload["freshness"]["is_stale"] is False
    assert payload["freshness"]["latest_verification_run_id"] is not None
    assert payload["latest_review_run"]["run_id"] == payload["freshness"]["latest_review_run_id"]
    assert payload["latest_review_run"]["status"] == "COMPLETED"
    assert payload["latest_review_run"]["remaining_flagged_claims"] == 4
    assert payload["previous_review_run"] is None
    assert payload["queue_state"] == {
        "draft_id": str(draft_id),
        "total_flagged_claims": 4,
        "resolved_flagged_claims": 0,
        "remaining_flagged_claims": 4,
        "next_claim_id": payload["active_queue_claims"][0]["claim_id"],
    }
    assert payload["active_queue_claims"][0]["draft_sequence"] == 2
    assert payload["active_queue_claims"][0]["assertion_context"] is not None
    assert payload["active_queue_claims"][0]["reasoning"] is not None
    assert payload["active_queue_claims"][0]["reasoning_categories"] == ["weak_support"]
    assert payload["active_queue_claims"][0]["changed_since_last_run"] is False
    assert payload["active_queue_claims"][0]["change_summary"]["changed_since_last_run"] is False
    assert payload["active_queue_claims"][0]["contradiction_flags"] == []
    assert payload["active_queue_claims"][0]["claim_relationships"] == []
    assert [claim["claim_text"] for claim in payload["active_queue_claims"]] == [
        "Doe reviewed the contract and approved the invoice.",
        "Doe signed the contract on March 1.",
        "Doe delivered the notice.",
        "Doe admitted the error.",
    ]
    assert payload["resolved_claims"] == []
    assert payload["flagged_claim_counts"] == {
        "unsupported": 2,
        "overstated": 1,
        "ambiguous": 1,
        "unverified": 0,
        "total": 4,
    }
    assert payload["review_overview"] == {
        "total_claims": 5,
        "total_flagged_claims": 4,
        "highest_severity_bucket": SupportStatus.UNSUPPORTED.value,
        "top_issue_categories": [
            "contextual_support_only",
            "missing_citation",
            "narrow_support",
        ],
    }
    assert [claim["claim_text"] for claim in payload["top_risky_claims"]] == [
        "Doe delivered the notice.",
        "Doe admitted the error.",
        "Doe reviewed the contract and approved the invoice.",
        "Doe signed the contract on March 1.",
    ]
    assert [claim["claim_text"] for claim in payload["issue_buckets"]["unsupported"]] == [
        "Doe delivered the notice.",
        "Doe admitted the error.",
    ]
    assert [claim["claim_text"] for claim in payload["issue_buckets"]["overstated"]] == [
        "Doe reviewed the contract and approved the invoice.",
    ]
    assert [claim["claim_text"] for claim in payload["issue_buckets"]["ambiguous"]] == [
        "Doe signed the contract on March 1.",
    ]
    assert payload["issue_buckets"]["unverified"] == []
    assert [bucket["flag"] for bucket in payload["flag_buckets"]] == [
        "contextual_support_only",
        "missing_citation",
        "narrow_support",
        "subject_mismatch",
    ]
    assert payload["intelligence_summary"] == {
        "risk_distribution": {
            "supported": 1,
            "partially_supported": 0,
            "overstated": 1,
            "ambiguous": 1,
            "unsupported": 2,
            "unverified": 0,
        },
        "most_unstable_claim_ids": [],
        "repeatedly_changed_claim_ids": [],
        "weak_support_claim_ids": [
            payload["active_queue_claims"][0]["claim_id"],
            payload["active_queue_claims"][1]["claim_id"],
        ],
        "contradiction_claim_ids": [],
        "contradiction_pair_count": 0,
        "duplicate_pair_count": 0,
        "weak_support_clusters": [
            {
                "flag": "contextual_support_only",
                "claim_count": 1,
                "claim_ids": [payload["active_queue_claims"][1]["claim_id"]],
            },
            {
                "flag": "narrow_support",
                "claim_count": 1,
                "claim_ids": [payload["active_queue_claims"][0]["claim_id"]],
            },
        ],
    }
    assert [claim["claim_text"] for claim in payload["flag_buckets"][0]["claims"]] == [
        "Doe signed the contract on March 1.",
    ]
    assert [claim["claim_text"] for claim in payload["flag_buckets"][2]["claims"]] == [
        "Doe reviewed the contract and approved the invoice.",
    ]
    assert "review summary" in payload["summary"].lower()


def test_review_endpoint_preserves_structured_support_in_review_buckets(
    client: TestClient,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_provider(monkeypatch, provider="openai", api_key="test-key")
    draft_id = _seed_reviewable_draft(session)

    response = client.post(f"/api/v1/drafts/{draft_id}/review")

    assert response.status_code == 200
    payload = response.json()
    assert payload["draft_id"] == str(draft_id)
    assert payload["total_claims"] == 5
    assert payload["flagged_claim_counts"] == {
        "unsupported": 2,
        "ambiguous": 1,
        "overstated": 0,
        "unverified": 0,
        "total": 3,
    }
    assert payload["review_overview"] == {
        "total_claims": 5,
        "total_flagged_claims": 3,
        "highest_severity_bucket": SupportStatus.UNSUPPORTED.value,
        "top_issue_categories": [
            "contextual_support_only",
            "missing_citation",
            "subject_mismatch",
        ],
    }
    assert [claim["claim_text"] for claim in payload["top_risky_claims"]] == [
        "Doe delivered the notice.",
        "Doe admitted the error.",
        "Doe signed the contract on March 1.",
    ]
    assert [claim["claim_text"] for claim in payload["issue_buckets"]["unsupported"]] == [
        "Doe delivered the notice.",
        "Doe admitted the error.",
    ]
    assert payload["issue_buckets"]["overstated"] == []
    assert [claim["claim_text"] for claim in payload["issue_buckets"]["ambiguous"]] == [
        "Doe signed the contract on March 1.",
    ]
    assert payload["issue_buckets"]["unverified"] == []
    assert [bucket["flag"] for bucket in payload["flag_buckets"]] == [
        "contextual_support_only",
        "missing_citation",
        "subject_mismatch",
    ]
    assert [claim["claim_text"] for claim in payload["flag_buckets"][0]["claims"]] == [
        "Doe signed the contract on March 1.",
    ]
    ambiguous_bucket_claim = payload["issue_buckets"]["ambiguous"][0]
    assert ambiguous_bucket_claim["primary_anchor"] == "p.30:1-30:2"
    assert [assessment["anchor"] for assessment in ambiguous_bucket_claim["support_assessments"]] == [
        "p.30:1-30:2",
        "p.30:3-30:4",
    ]
    assert "review summary" in payload["summary"].lower()


def test_review_state_endpoint_reads_persisted_queue_without_recomputation(
    client: TestClient,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_provider(monkeypatch, provider=None, api_key=None)
    draft_id = _seed_reviewable_draft(session)

    fresh_review_response = client.post(f"/api/v1/drafts/{draft_id}/review")

    assert fresh_review_response.status_code == 200

    def fail_compile(*args, **kwargs):
        raise AssertionError("compile_draft should not run for GET /review-state")

    monkeypatch.setattr("app.api.v1.drafts.DraftCompileService.compile_draft", fail_compile)

    review_state_response = client.get(f"/api/v1/drafts/{draft_id}/review-state")

    assert review_state_response.status_code == 200
    payload = review_state_response.json()
    assert payload["draft_id"] == str(draft_id)
    assert payload["freshness"]["state_source"] == "persisted_read"
    assert payload["freshness"]["has_persisted_review_runs"] is True
    assert payload["freshness"]["last_review_run_at"] is not None
    assert payload["freshness"]["latest_review_run_id"] is not None
    assert payload["freshness"]["latest_review_run_status"] == "COMPLETED"
    assert payload["freshness"]["latest_decision_at"] is None
    assert payload["freshness"]["has_decisions_after_latest_run"] is False
    assert payload["freshness"]["is_stale"] is False
    assert payload["freshness"]["latest_verification_run_id"] is not None
    assert payload["latest_review_run"]["run_id"] == payload["freshness"]["latest_review_run_id"]
    assert payload["previous_review_run"] is None
    assert payload["queue_state"] == {
        "draft_id": str(draft_id),
        "total_flagged_claims": 4,
        "resolved_flagged_claims": 0,
        "remaining_flagged_claims": 4,
        "next_claim_id": payload["active_queue_claims"][0]["claim_id"],
    }
    assert [claim["claim_text"] for claim in payload["active_queue_claims"]] == [
        "Doe reviewed the contract and approved the invoice.",
        "Doe signed the contract on March 1.",
        "Doe delivered the notice.",
        "Doe admitted the error.",
    ]
    assert payload["resolved_claims"] == []


@pytest.mark.parametrize(
    ("action", "decision_body", "expected_note", "expected_replacement"),
    [
        ("acknowledge_risk", {}, None, None),
        (
            "mark_for_revision",
            {"note": "Split the actor/date mismatch before final review."},
            "Split the actor/date mismatch before final review.",
            None,
        ),
        (
            "resolve_with_edit",
            {
                "note": "Narrowed to the supported date.",
                "proposed_replacement_text": "Doe delivered the notice on March 3.",
            },
            "Narrowed to the supported date.",
            "Doe delivered the notice on March 3.",
        ),
    ],
)
def test_review_state_endpoint_projects_latest_decisions_without_recomputation(
    client: TestClient,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    decision_body: dict[str, str],
    expected_note: str | None,
    expected_replacement: str | None,
) -> None:
    _configure_provider(monkeypatch, provider=None, api_key=None)
    draft_id = _seed_reviewable_draft(session)

    fresh_review_response = client.post(f"/api/v1/drafts/{draft_id}/review")
    assert fresh_review_response.status_code == 200

    target_claim = session.query(ClaimUnit).filter(ClaimUnit.text == "Doe delivered the notice.").one()
    decision_response = client.post(
        f"/api/v1/claims/{target_claim.id}/decisions",
        json={"action": action, **decision_body},
    )
    assert decision_response.status_code == 201

    def fail_compile(*args, **kwargs):
        raise AssertionError("compile_draft should not run for GET /review-state")

    monkeypatch.setattr("app.api.v1.drafts.DraftCompileService.compile_draft", fail_compile)

    review_state_response = client.get(f"/api/v1/drafts/{draft_id}/review-state")

    assert review_state_response.status_code == 200
    payload = review_state_response.json()
    assert payload["freshness"]["state_source"] == "persisted_read"
    assert payload["freshness"]["latest_review_run_id"] is not None
    assert payload["freshness"]["latest_review_run_status"] == "COMPLETED"
    assert payload["freshness"]["latest_decision_at"] is not None
    assert payload["freshness"]["has_decisions_after_latest_run"] is True
    assert payload["freshness"]["is_stale"] is True
    assert payload["queue_state"] == {
        "draft_id": str(draft_id),
        "total_flagged_claims": 4,
        "resolved_flagged_claims": 1,
        "remaining_flagged_claims": 3,
        "next_claim_id": payload["active_queue_claims"][0]["claim_id"],
    }
    assert [claim["claim_text"] for claim in payload["active_queue_claims"]] == [
        "Doe reviewed the contract and approved the invoice.",
        "Doe signed the contract on March 1.",
        "Doe admitted the error.",
    ]
    assert [claim["claim"]["claim_text"] for claim in payload["resolved_claims"]] == [
        "Doe delivered the notice.",
    ]
    assert payload["resolved_claims"][0]["latest_decision"] == {
        "action": action,
        "note": expected_note,
        "proposed_replacement_text": expected_replacement,
        "created_at": payload["resolved_claims"][0]["latest_decision"]["created_at"],
    }


@pytest.mark.parametrize(
    ("action", "body", "expected_note", "expected_replacement"),
    [
        ("acknowledge_risk", {}, None, None),
        (
            "mark_for_revision",
            {"note": "Split the actor/date mismatch before final review."},
            "Split the actor/date mismatch before final review.",
            None,
        ),
        (
            "resolve_with_edit",
            {
                "note": "Narrowed to the supported date.",
                "proposed_replacement_text": "Doe delivered the notice on March 3.",
            },
            "Narrowed to the supported date.",
            "Doe delivered the notice on March 3.",
        ),
    ],
)
def test_review_endpoint_projects_latest_persisted_decisions_into_queue_state(
    client: TestClient,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    body: dict[str, str],
    expected_note: str | None,
    expected_replacement: str | None,
) -> None:
    _configure_provider(monkeypatch, provider=None, api_key=None)
    draft_id = _seed_reviewable_draft(session)
    target_claim = session.query(ClaimUnit).filter(ClaimUnit.text == "Doe delivered the notice.").one()

    decision_response = client.post(
        f"/api/v1/claims/{target_claim.id}/decisions",
        json={"action": action, **body},
    )
    review_response = client.post(f"/api/v1/drafts/{draft_id}/review")

    assert decision_response.status_code == 201
    assert review_response.status_code == 200
    payload = review_response.json()
    assert payload["freshness"]["latest_review_run_id"] is not None
    assert payload["freshness"]["latest_review_run_status"] == "COMPLETED"
    assert payload["queue_state"]["draft_id"] == str(draft_id)
    assert payload["queue_state"]["resolved_flagged_claims"] == 1
    assert payload["queue_state"]["remaining_flagged_claims"] == len(payload["active_queue_claims"])
    assert payload["queue_state"]["total_flagged_claims"] == (
        len(payload["active_queue_claims"]) + len(payload["resolved_claims"])
    )
    assert [claim["claim"]["claim_text"] for claim in payload["resolved_claims"]] == [
        "Doe delivered the notice.",
    ]
    assert "Doe delivered the notice." not in [claim["claim_text"] for claim in payload["active_queue_claims"]]
    assert payload["resolved_claims"][0]["latest_decision"] == {
        "action": action,
        "note": expected_note,
        "proposed_replacement_text": expected_replacement,
        "created_at": payload["resolved_claims"][0]["latest_decision"]["created_at"],
    }
    assert payload["flagged_claim_counts"]["total"] == len(payload["active_queue_claims"])


def test_review_endpoint_persists_previous_run_metadata_on_subsequent_execution(
    client: TestClient,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_provider(monkeypatch, provider=None, api_key=None)
    draft_id = _seed_reviewable_draft(session)

    first_review = client.post(f"/api/v1/drafts/{draft_id}/review")
    second_review = client.post(f"/api/v1/drafts/{draft_id}/review")

    assert first_review.status_code == 200
    assert second_review.status_code == 200
    payload = second_review.json()
    assert payload["latest_review_run"] is not None
    assert payload["previous_review_run"] is not None
    assert payload["latest_review_run"]["run_id"] != payload["previous_review_run"]["run_id"]
    assert payload["latest_review_run"]["status"] == "COMPLETED"
    assert payload["previous_review_run"]["status"] == "COMPLETED"
    assert payload["latest_review_run"]["remaining_flagged_claims"] == payload["queue_state"]["remaining_flagged_claims"]


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


def _seed_reviewable_draft(session: Session) -> uuid.UUID:
    matter = Matter(name="Acme v. Doe", status="ACTIVE")
    source_document = SourceDocument(
        matter=matter,
        file_name="deposition.txt",
        source_type=SourceType.DEPOSITION,
        raw_file_path="/tmp/deposition.txt",
        parser_status=ParserStatus.COMPLETED,
        parser_confidence=0.95,
    )
    draft = Draft(matter=matter, title="Motion draft", mode=DraftMode.COMPILE)
    session.add_all([matter, source_document, draft])
    session.flush()

    _add_claim(
        session,
        draft=draft,
        source_document=source_document,
        paragraph_index=1,
        claim_text="Smith signed the contract.",
        support_rows=[
            {
                "raw_text": "A. Smith signed the contract.",
                "page_start": 10,
                "line_start": 1,
                "page_end": 10,
                "line_end": 2,
                "speaker": "A",
                "segment_type": "ANSWER_BLOCK",
                "sequence_order": 1,
            }
        ],
    )
    _add_claim(
        session,
        draft=draft,
        source_document=source_document,
        paragraph_index=2,
        claim_text="Doe reviewed the contract and approved the invoice.",
        support_rows=[
            {
                "raw_text": "A. Doe reviewed the contract.",
                "page_start": 20,
                "line_start": 1,
                "page_end": 20,
                "line_end": 2,
                "speaker": "A",
                "segment_type": "ANSWER_BLOCK",
                "sequence_order": 1,
            }
        ],
    )
    _add_claim(
        session,
        draft=draft,
        source_document=source_document,
        paragraph_index=3,
        claim_text="Doe signed the contract on March 1.",
        support_rows=[
            {
                "raw_text": "Q. Did Doe sign the contract on March 1?",
                "page_start": 30,
                "line_start": 1,
                "page_end": 30,
                "line_end": 2,
                "speaker": "Q",
                "segment_type": "QUESTION_BLOCK",
                "sequence_order": 1,
            },
            {
                "raw_text": "A. I do not remember whether Doe signed it.",
                "page_start": 30,
                "line_start": 3,
                "page_end": 30,
                "line_end": 4,
                "speaker": "A",
                "segment_type": "ANSWER_BLOCK",
                "sequence_order": 2,
            },
        ],
    )
    _add_claim(
        session,
        draft=draft,
        source_document=source_document,
        paragraph_index=4,
        claim_text="Doe delivered the notice.",
        support_rows=[
            {
                "raw_text": "A. Smith reviewed the invoice.",
                "page_start": 40,
                "line_start": 1,
                "page_end": 40,
                "line_end": 2,
                "speaker": "A",
                "segment_type": "ANSWER_BLOCK",
                "sequence_order": 1,
            }
        ],
    )
    _add_claim(
        session,
        draft=draft,
        source_document=source_document,
        paragraph_index=5,
        claim_text="Doe admitted the error.",
        support_rows=[],
    )

    session.commit()
    return draft.id


def _add_claim(
    session: Session,
    *,
    draft: Draft,
    source_document: SourceDocument,
    paragraph_index: int,
    claim_text: str,
    support_rows: list[dict[str, object]],
) -> ClaimUnit:
    assertion = Assertion(
        draft=draft,
        paragraph_index=paragraph_index,
        sentence_index=1,
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
    session.add_all([assertion, claim])
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

    session.flush()
    return claim
