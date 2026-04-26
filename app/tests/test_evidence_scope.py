import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.enums import ClaimType, DraftMode, ParserStatus, SourceType, SupportStatus
from app.models import Assertion, Base, ClaimUnit, Draft, Matter, SourceDocument
from app.repositories.drafts import DraftRepository
from app.repositories.evidence_bundles import EvidenceBundleRepository
from app.schemas.draft import DraftCreate
from app.schemas.evidence_bundle import EvidenceBundleCreate
from app.services.verification.classifier import VerificationExecution, VerificationResult
from app.services.workflows.draft_compile import DraftCompileService
from app.services.workflows.draft_review import DraftReviewService


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


def test_evidence_bundle_scopes_draft_to_explicit_source_documents(session: Session) -> None:
    matter = Matter(name="Acme v. Doe", status="ACTIVE")
    first_source = SourceDocument(
        matter=matter,
        file_name="a-declaration.txt",
        source_type=SourceType.DECLARATION,
        raw_file_path="/tmp/a-declaration.txt",
        parser_status=ParserStatus.COMPLETED,
        parser_confidence=0.85,
    )
    second_source = SourceDocument(
        matter=matter,
        file_name="b-deposition.txt",
        source_type=SourceType.DEPOSITION,
        raw_file_path="/tmp/b-deposition.txt",
        parser_status=ParserStatus.COMPLETED,
        parser_confidence=0.95,
    )
    unscoped_source = SourceDocument(
        matter=matter,
        file_name="c-exhibit.txt",
        source_type=SourceType.EXHIBIT,
        raw_file_path="/tmp/c-exhibit.txt",
        parser_status=ParserStatus.COMPLETED,
        parser_confidence=0.8,
    )
    session.add_all([matter, first_source, second_source, unscoped_source])
    session.flush()

    bundles = EvidenceBundleRepository(session)
    bundle = bundles.create(
        matter.id,
        EvidenceBundleCreate(
            name="Initial Record",
            source_document_ids=[second_source.id, first_source.id],
        ),
    )
    draft = DraftRepository(session).create(
        matter.id,
        DraftCreate(
            title="Scoped draft",
            mode=DraftMode.COMPILE,
            evidence_bundle_id=bundle.id,
        ),
    )
    session.commit()

    bundle_source_ids = {source.id for source in bundles.list_source_documents(bundle.id)}
    resolved_source_ids = bundles.resolve_allowed_source_document_ids_for_draft(draft.id)

    assert draft.evidence_bundle_id == bundle.id
    assert bundle_source_ids == {first_source.id, second_source.id}
    assert set(resolved_source_ids) == {first_source.id, second_source.id}
    assert unscoped_source.id not in resolved_source_ids


def test_evidence_scope_falls_back_to_all_matter_sources_for_unscoped_drafts(session: Session) -> None:
    matter = Matter(name="Acme v. Doe", status="ACTIVE")
    first_source = SourceDocument(
        matter=matter,
        file_name="declaration.txt",
        source_type=SourceType.DECLARATION,
        raw_file_path="/tmp/declaration.txt",
        parser_status=ParserStatus.COMPLETED,
        parser_confidence=0.85,
    )
    second_source = SourceDocument(
        matter=matter,
        file_name="deposition.txt",
        source_type=SourceType.DEPOSITION,
        raw_file_path="/tmp/deposition.txt",
        parser_status=ParserStatus.COMPLETED,
        parser_confidence=0.95,
    )
    session.add_all([matter, first_source, second_source])
    session.flush()

    draft = DraftRepository(session).create(
        matter.id,
        DraftCreate(
            title="Unscoped draft",
            mode=DraftMode.DRAFT,
        ),
    )
    session.commit()

    bundles = EvidenceBundleRepository(session)

    assert set(bundles.resolve_allowed_source_document_ids_for_draft(draft.id)) == {
        first_source.id,
        second_source.id,
    }
    assert bundles.resolve_allowed_source_document_ids_for_draft(draft.id, fallback_to_matter_sources=False) == []


class StubVerificationService:
    def __init__(self, executions: dict[uuid.UUID, VerificationExecution]) -> None:
        self.executions = executions

    def verify_claim(
        self,
        claim_id: uuid.UUID,
        *,
        model_version: str | None = None,
        prompt_version: str | None = None,
    ) -> VerificationExecution:
        del model_version, prompt_version
        return self.executions[claim_id]


def test_scoped_draft_compile_and_review_workflows_remain_compatible(session: Session) -> None:
    matter = Matter(name="Acme v. Doe", status="ACTIVE")
    scoped_source = SourceDocument(
        matter=matter,
        file_name="scoped-deposition.txt",
        source_type=SourceType.DEPOSITION,
        raw_file_path="/tmp/scoped-deposition.txt",
        parser_status=ParserStatus.COMPLETED,
        parser_confidence=0.95,
    )
    unscoped_source = SourceDocument(
        matter=matter,
        file_name="unscoped-exhibit.txt",
        source_type=SourceType.EXHIBIT,
        raw_file_path="/tmp/unscoped-exhibit.txt",
        parser_status=ParserStatus.COMPLETED,
        parser_confidence=0.8,
    )
    session.add_all([matter, scoped_source, unscoped_source])
    session.flush()

    bundle = EvidenceBundleRepository(session).create(
        matter.id,
        EvidenceBundleCreate(
            name="Deposition Only",
            source_document_ids=[scoped_source.id],
        ),
    )
    draft = Draft(
        matter=matter,
        evidence_bundle_id=bundle.id,
        title="Compile draft",
        mode=DraftMode.COMPILE,
    )
    assertion = Assertion(
        draft=draft,
        paragraph_index=1,
        sentence_index=1,
        raw_text="Doe delivered the notice.",
        normalized_text="doe delivered the notice.",
    )
    claim = ClaimUnit(
        assertion=assertion,
        text="Doe delivered the notice.",
        normalized_text="doe delivered the notice.",
        claim_type=ClaimType.FACT,
        sequence_order=1,
        support_status=SupportStatus.UNVERIFIED,
    )
    session.add_all([draft, assertion, claim])
    session.commit()

    execution = VerificationExecution(
        run=object(),
        result=VerificationResult(
            model_version="stub",
            prompt_version="test",
            deterministic_flags=["subject_mismatch"],
            verdict=SupportStatus.UNSUPPORTED,
            reasoning="Scoped draft verification still runs.",
            suggested_fix="Relink the claim to in-scope record support.",
            confidence_score=0.25,
            primary_anchor="p.12:1-12:2",
            support_assessments=[],
        ),
    )
    compile_result = DraftCompileService(
        session,
        verification_service=StubVerificationService({claim.id: execution}),
    ).compile_draft(draft.id, max_concurrency=1)
    review_result = DraftReviewService().review_draft(compile_result)

    assert draft.evidence_bundle_id == bundle.id
    assert compile_result.total_claims == 1
    assert compile_result.counts.unsupported == 1
    assert [flagged.claim_text for flagged in compile_result.flagged_claims] == [
        "Doe delivered the notice.",
    ]
    assert review_result.flagged_claim_counts.unsupported == 1
    assert [flagged.claim_text for flagged in review_result.issue_buckets.unsupported] == [
        "Doe delivered the notice.",
    ]
    assert review_result.top_risky_claims[0].primary_anchor == "p.12:1-12:2"
