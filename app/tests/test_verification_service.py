import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.core.provenance import build_exhibit_page_anchor_metadata, build_paragraph_anchor_metadata
from app.core.enums import ClaimType, DraftMode, LinkType, ParserStatus, SourceType, SupportStatus
from app.models import (
    CURRENT_SUPPORT_SNAPSHOT_VERSION,
    Assertion,
    Base,
    ClaimUnit,
    Draft,
    Matter,
    Segment,
    SourceDocument,
    SupportLink,
)
from app.repositories.claims import ClaimsRepository
from app.repositories.evidence_bundles import EvidenceBundleRepository
from app.repositories.verification import VerificationRepository
from app.schemas.evidence_bundle import EvidenceBundleCreate
from app.services.llm.base import (
    ProviderRequest,
    ProviderResponse,
    ProviderSupportAssessment,
    VerificationProvider,
)
from app.services.parsing.normalization import normalize_for_match
from app.services.verification.classifier import ClaimVerificationService, VerificationClassifier


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


class InspectingProvider(VerificationProvider):
    name = "inspecting"

    def __init__(self, observed_session: Session) -> None:
        self.model_version = "inspect-v1"
        self.observed_session = observed_session
        self.session_in_transaction_during_call: bool | None = None
        self.last_request: ProviderRequest | None = None

    def verify(self, request: ProviderRequest) -> ProviderResponse:
        self.session_in_transaction_during_call = bool(self.observed_session.in_transaction())
        self.last_request = request
        return ProviderResponse(
            verdict=SupportStatus.SUPPORTED,
            reasoning="Provider supports the claim.",
            suggested_fix=None,
            confidence_score=0.8,
            support_assessments=[
                ProviderSupportAssessment(
                    segment_id=request.support_items[0].segment_id,
                    anchor=request.support_items[0].anchor,
                    role="primary",
                    contribution="Direct answer testimony states the proposition.",
                )
            ],
            primary_anchor=request.support_items[0].anchor,
        )


class InspectingClassifier(VerificationClassifier):
    def __init__(self, provider: InspectingProvider) -> None:
        super().__init__(settings=Settings())
        self.provider = provider

    def _build_provider(self, *, model_version: str):
        return self.provider


def test_verification_service_separates_db_phases_and_persists_run(session: Session) -> None:
    claim_id = _seed_claim_with_support(session)
    provider = InspectingProvider(session)
    classifier = InspectingClassifier(provider)

    execution = ClaimVerificationService(session, classifier=classifier).verify_claim(claim_id)

    assert provider.session_in_transaction_during_call is False
    assert execution.result.verdict == SupportStatus.SUPPORTED
    assert execution.result.primary_anchor == "p.10:3-10:4"
    assert [assessment.role for assessment in execution.result.support_assessments] == ["primary"]

    session.expire_all()
    persisted_claim = ClaimsRepository(session).get(claim_id)
    persisted_runs = VerificationRepository(session).list_by_claim(claim_id)

    assert persisted_claim is not None
    assert persisted_claim.support_status == SupportStatus.SUPPORTED
    assert len(persisted_runs) == 1
    assert persisted_runs[0].id == execution.run.id
    assert persisted_runs[0].verdict == SupportStatus.SUPPORTED
    assert persisted_runs[0].support_snapshot_version == CURRENT_SUPPORT_SNAPSHOT_VERSION
    assert persisted_runs[0].support_snapshot is not None
    assert persisted_runs[0].support_snapshot["claim_scope"]["scope_kind"] == "matter_fallback"
    assert persisted_runs[0].support_snapshot["provider_output"]["primary_anchor"] == "p.10:3-10:4"
    assert persisted_runs[0].support_snapshot["provider_output"]["support_assessments"] == [
        {
            "segment_id": str(provider.last_request.support_items[0].segment_id),
            "anchor": "p.10:3-10:4",
            "role": "primary",
            "contribution": "Direct answer testimony states the proposition.",
        }
    ]
    assert persisted_runs[0].support_snapshot["support_items"][0]["anchor"] == "p.10:3-10:4"
    assert persisted_runs[0].support_snapshot["valid_support_links"][0]["anchor"] == "p.10:3-10:4"


def test_verification_service_accepts_declaration_paragraph_provenance(session: Session) -> None:
    claim_id = _seed_claim_with_declaration_support(session)
    provider = InspectingProvider(session)
    classifier = InspectingClassifier(provider)

    execution = ClaimVerificationService(session, classifier=classifier).verify_claim(claim_id)

    assert provider.session_in_transaction_during_call is False
    assert provider.last_request is not None
    assert provider.last_request.support_items[0].anchor == "¶2"
    assert execution.result.primary_anchor == "¶2"
    assert execution.result.verdict == SupportStatus.SUPPORTED

    session.expire_all()
    persisted_runs = VerificationRepository(session).list_by_claim(claim_id)
    assert len(persisted_runs) == 1
    assert persisted_runs[0].verdict == SupportStatus.SUPPORTED


def test_verification_service_accepts_exhibit_page_provenance(session: Session) -> None:
    claim_id = _seed_claim_with_exhibit_support(session)
    provider = InspectingProvider(session)
    classifier = InspectingClassifier(provider)

    execution = ClaimVerificationService(session, classifier=classifier).verify_claim(claim_id)

    assert provider.session_in_transaction_during_call is False
    assert provider.last_request is not None
    assert provider.last_request.support_items[0].anchor == "ex. p.1 block 2"
    assert execution.result.primary_anchor == "ex. p.1 block 2"
    assert execution.result.verdict == SupportStatus.SUPPORTED

    session.expire_all()
    persisted_runs = VerificationRepository(session).list_by_claim(claim_id)
    assert len(persisted_runs) == 1
    assert persisted_runs[0].verdict == SupportStatus.SUPPORTED


def test_verification_service_excludes_invalid_existing_links_from_provider_reasoning(session: Session) -> None:
    claim_id, valid_anchor = _seed_claim_with_scoped_and_out_of_scope_links(session)
    provider = InspectingProvider(session)
    classifier = InspectingClassifier(provider)

    execution = ClaimVerificationService(session, classifier=classifier).verify_claim(claim_id)

    assert provider.session_in_transaction_during_call is False
    assert provider.last_request is not None
    assert [item.anchor for item in provider.last_request.support_items] == [valid_anchor]
    assert "out_of_scope_support_link" in execution.result.deterministic_flags
    assert execution.result.primary_anchor == valid_anchor
    assert "Excluded 1 invalid support link(s) from reasoning" in execution.result.reasoning
    assert execution.result.verdict == SupportStatus.SUPPORTED

    persisted_run = VerificationRepository(session).list_by_claim(claim_id)[0]
    assert persisted_run.support_snapshot_version == CURRENT_SUPPORT_SNAPSHOT_VERSION
    assert persisted_run.support_snapshot is not None
    assert persisted_run.support_snapshot["claim_scope"]["scope_kind"] == "bundle"
    assert persisted_run.support_snapshot["claim_scope"]["evidence_bundle_id"] is not None
    assert len(persisted_run.support_snapshot["valid_support_links"]) == 1
    assert len(persisted_run.support_snapshot["excluded_support_links"]) == 1
    assert persisted_run.support_snapshot["excluded_support_links"][0]["code"] == "out_of_scope_support_link"
    assert persisted_run.support_snapshot["provider_output"]["primary_anchor"] == valid_anchor


def test_verification_service_surfaces_invalid_existing_links_when_no_valid_support_remains(session: Session) -> None:
    claim_id = _seed_claim_with_only_out_of_scope_link(session)
    provider = InspectingProvider(session)
    classifier = InspectingClassifier(provider)

    execution = ClaimVerificationService(session, classifier=classifier).verify_claim(claim_id)

    assert provider.last_request is None
    assert "missing_citation" in execution.result.deterministic_flags
    assert "out_of_scope_support_link" in execution.result.deterministic_flags
    assert execution.result.verdict == SupportStatus.UNSUPPORTED
    assert execution.result.primary_anchor is None
    assert execution.result.support_assessments == []
    assert "No valid support links remain after excluding invalid existing links." in execution.result.reasoning

    persisted_run = VerificationRepository(session).list_by_claim(claim_id)[0]
    assert persisted_run.support_snapshot_version == CURRENT_SUPPORT_SNAPSHOT_VERSION
    assert persisted_run.support_snapshot is not None
    assert persisted_run.support_snapshot["valid_support_links"] == []
    assert len(persisted_run.support_snapshot["excluded_support_links"]) == 1
    assert persisted_run.support_snapshot["excluded_support_links"][0]["code"] == "out_of_scope_support_link"
    assert persisted_run.support_snapshot["support_items"] == []
    assert persisted_run.support_snapshot["provider_output"]["primary_anchor"] is None
    assert persisted_run.support_snapshot["provider_output"]["support_assessments"] == []


def _seed_claim_with_support(session: Session) -> uuid.UUID:
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
        raw_text="Smith signed the contract.",
        normalized_text=normalize_for_match("Smith signed the contract."),
    )
    claim = ClaimUnit(
        assertion=assertion,
        text="Smith signed the contract.",
        normalized_text=normalize_for_match("Smith signed the contract."),
        claim_type=ClaimType.FACT,
        sequence_order=1,
        support_status=SupportStatus.UNVERIFIED,
    )
    session.add_all([matter, source_document, draft, assertion, claim])
    session.flush()

    segment = Segment(
        id=uuid.uuid4(),
        source_document_id=source_document.id,
        page_start=10,
        line_start=3,
        page_end=10,
        line_end=4,
        raw_text="A. Smith signed the contract on March 1.",
        normalized_text=normalize_for_match("A. Smith signed the contract on March 1."),
        speaker="A",
        segment_type="ANSWER_BLOCK",
    )
    session.add(segment)
    session.flush()
    session.add(
        SupportLink(
            id=uuid.uuid4(),
            claim_unit_id=claim.id,
            segment_id=segment.id,
            sequence_order=1,
            link_type=LinkType.MANUAL,
            citation_text=None,
            user_confirmed=True,
        )
    )
    session.commit()
    return claim.id


def _seed_claim_with_declaration_support(session: Session) -> uuid.UUID:
    matter = Matter(name="Acme v. Doe", status="ACTIVE")
    source_document = SourceDocument(
        matter=matter,
        file_name="declaration.txt",
        source_type=SourceType.DECLARATION,
        raw_file_path="/tmp/declaration.txt",
        parser_status=ParserStatus.COMPLETED,
        parser_confidence=0.85,
    )
    draft = Draft(matter=matter, title="Motion draft", mode=DraftMode.DRAFT)
    assertion = Assertion(
        draft=draft,
        raw_text="I signed the declaration.",
        normalized_text=normalize_for_match("I signed the declaration."),
    )
    claim = ClaimUnit(
        assertion=assertion,
        text="I signed the declaration.",
        normalized_text=normalize_for_match("I signed the declaration."),
        claim_type=ClaimType.FACT,
        sequence_order=1,
        support_status=SupportStatus.UNVERIFIED,
    )
    session.add_all([matter, source_document, draft, assertion, claim])
    session.flush()

    segment = Segment(
        id=uuid.uuid4(),
        source_document_id=source_document.id,
        page_start=None,
        line_start=None,
        page_end=None,
        line_end=None,
        anchor_metadata=build_paragraph_anchor_metadata(paragraph_number=2, sequence_order=2),
        raw_text="2. I signed the declaration.",
        normalized_text=normalize_for_match("2. I signed the declaration."),
        speaker="DECLARANT ¶2",
        segment_type="DECLARATION_PARAGRAPH",
    )
    session.add(segment)
    session.flush()
    session.add(
        SupportLink(
            id=uuid.uuid4(),
            claim_unit_id=claim.id,
            segment_id=segment.id,
            sequence_order=1,
            link_type=LinkType.MANUAL,
            citation_text=None,
            user_confirmed=True,
        )
    )
    session.commit()
    return claim.id


def _seed_claim_with_exhibit_support(session: Session) -> uuid.UUID:
    matter = Matter(name="Acme v. Doe", status="ACTIVE")
    source_document = SourceDocument(
        matter=matter,
        file_name="exhibit.txt",
        source_type=SourceType.EXHIBIT,
        raw_file_path="/tmp/exhibit.txt",
        parser_status=ParserStatus.COMPLETED,
        parser_confidence=0.8,
    )
    draft = Draft(matter=matter, title="Motion draft", mode=DraftMode.DRAFT)
    assertion = Assertion(
        draft=draft,
        raw_text="The contract was signed on March 1.",
        normalized_text=normalize_for_match("The contract was signed on March 1."),
    )
    claim = ClaimUnit(
        assertion=assertion,
        text="The contract was signed on March 1.",
        normalized_text=normalize_for_match("The contract was signed on March 1."),
        claim_type=ClaimType.FACT,
        sequence_order=1,
        support_status=SupportStatus.UNVERIFIED,
    )
    session.add_all([matter, source_document, draft, assertion, claim])
    session.flush()

    segment = Segment(
        id=uuid.uuid4(),
        source_document_id=source_document.id,
        page_start=None,
        line_start=None,
        page_end=None,
        line_end=None,
        anchor_metadata=build_exhibit_page_anchor_metadata(page_number=1, block_index=2, sequence_order=2),
        raw_text="The contract was signed on March 1.",
        normalized_text=normalize_for_match("The contract was signed on March 1."),
        speaker=None,
        segment_type="EXHIBIT_PAGE_BLOCK",
    )
    session.add(segment)
    session.flush()
    session.add(
        SupportLink(
            id=uuid.uuid4(),
            claim_unit_id=claim.id,
            segment_id=segment.id,
            sequence_order=1,
            link_type=LinkType.MANUAL,
            citation_text=None,
            user_confirmed=True,
        )
    )
    session.commit()
    return claim.id


def _seed_claim_with_scoped_and_out_of_scope_links(session: Session) -> tuple[uuid.UUID, str]:
    matter = Matter(name="Acme v. Doe", status="ACTIVE")
    in_scope_source = SourceDocument(
        matter=matter,
        file_name="deposition.txt",
        source_type=SourceType.DEPOSITION,
        raw_file_path="/tmp/deposition.txt",
        parser_status=ParserStatus.COMPLETED,
        parser_confidence=0.95,
    )
    out_of_scope_source = SourceDocument(
        matter=matter,
        file_name="exhibit.txt",
        source_type=SourceType.EXHIBIT,
        raw_file_path="/tmp/exhibit.txt",
        parser_status=ParserStatus.COMPLETED,
        parser_confidence=0.8,
    )
    session.add_all([matter, in_scope_source, out_of_scope_source])
    session.flush()

    evidence_bundle = EvidenceBundleRepository(session).create(
        matter.id,
        EvidenceBundleCreate(
            name="Deposition Only",
            source_document_ids=[in_scope_source.id],
        ),
    )
    draft = Draft(matter=matter, evidence_bundle_id=evidence_bundle.id, title="Motion draft", mode=DraftMode.DRAFT)
    assertion = Assertion(
        draft=draft,
        raw_text="Smith signed the contract.",
        normalized_text=normalize_for_match("Smith signed the contract."),
    )
    claim = ClaimUnit(
        assertion=assertion,
        text="Smith signed the contract.",
        normalized_text=normalize_for_match("Smith signed the contract."),
        claim_type=ClaimType.FACT,
        sequence_order=1,
        support_status=SupportStatus.UNVERIFIED,
    )
    session.add_all([draft, assertion, claim])
    session.flush()

    valid_segment = Segment(
        id=uuid.uuid4(),
        source_document_id=in_scope_source.id,
        page_start=10,
        line_start=3,
        page_end=10,
        line_end=4,
        raw_text="A. Smith signed the contract on March 1.",
        normalized_text=normalize_for_match("A. Smith signed the contract on March 1."),
        speaker="A",
        segment_type="ANSWER_BLOCK",
    )
    invalid_segment = Segment(
        id=uuid.uuid4(),
        source_document_id=out_of_scope_source.id,
        page_start=None,
        line_start=None,
        page_end=None,
        line_end=None,
        anchor_metadata=build_exhibit_page_anchor_metadata(page_number=1, block_index=1, sequence_order=1),
        raw_text="Subject: Contract Status",
        normalized_text=normalize_for_match("Subject: Contract Status"),
        speaker="Subject",
        segment_type="EXHIBIT_LABELED_BLOCK",
    )
    session.add_all([valid_segment, invalid_segment])
    session.flush()

    session.add_all(
        [
            SupportLink(
                id=uuid.uuid4(),
                claim_unit_id=claim.id,
                segment_id=valid_segment.id,
                sequence_order=1,
                link_type=LinkType.MANUAL,
                citation_text=None,
                user_confirmed=True,
            ),
            SupportLink(
                id=uuid.uuid4(),
                claim_unit_id=claim.id,
                segment_id=invalid_segment.id,
                sequence_order=2,
                link_type=LinkType.MANUAL,
                citation_text=None,
                user_confirmed=True,
            ),
        ]
    )
    session.commit()
    return claim.id, "p.10:3-10:4"


def _seed_claim_with_only_out_of_scope_link(session: Session) -> uuid.UUID:
    claim_id, _ = _seed_claim_with_scoped_and_out_of_scope_links(session)
    claim = ClaimsRepository(session).get(claim_id)
    assert claim is not None
    valid_link = next(
        link
        for link in claim.support_links
        if link.segment is not None and link.segment.source_document.source_type == SourceType.DEPOSITION
    )
    session.delete(valid_link)
    session.commit()
    return claim_id
