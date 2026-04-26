import uuid
import threading
import time

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import get_settings
from app.core.enums import ClaimType, DraftMode, LinkType, ParserStatus, SourceType, SupportStatus
from app.models import Assertion, Base, ClaimUnit, Draft, Matter, Segment, SourceDocument, SupportLink, VerificationRun
from app.services.llm.base import ProviderSupportAssessment
from app.services.parsing.normalization import normalize_for_match
from app.services.verification.classifier import VerificationExecution, VerificationResult
from app.services.workflows.draft_compile import DraftCompileService


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


def test_compile_draft_aggregates_mixed_verification_outcomes(
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_provider(monkeypatch, provider=None, api_key=None)
    draft_id = _seed_mixed_outcome_draft(session)

    result = DraftCompileService(session).compile_draft(draft_id, max_concurrency=1)

    assert result.total_claims == 5
    assert result.counts.supported == 1
    assert result.counts.partially_supported == 0
    assert result.counts.overstated == 1
    assert result.counts.ambiguous == 1
    assert result.counts.unsupported == 2
    assert result.counts.unverified == 0
    assert [claim.verdict for claim in result.flagged_claims] == [
        SupportStatus.OVERSTATED,
        SupportStatus.AMBIGUOUS,
        SupportStatus.UNSUPPORTED,
        SupportStatus.UNSUPPORTED,
    ]
    assert [claim.claim_text for claim in result.flagged_claims] == [
        "Doe reviewed the contract and approved the invoice.",
        "Doe signed the contract.",
        "Doe delivered the notice.",
        "Doe admitted the error.",
    ]


def test_compile_draft_preserves_provider_structured_output(
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_provider(monkeypatch, provider="openai", api_key="test-key")
    draft_id = _seed_provider_backed_draft(session)

    result = DraftCompileService(session).compile_draft(draft_id, max_concurrency=1)

    assert result.total_claims == 2
    assert result.counts.supported == 1
    assert result.counts.ambiguous == 1
    assert len(result.flagged_claims) == 1

    flagged_claim = result.flagged_claims[0]
    assert flagged_claim.verdict == SupportStatus.AMBIGUOUS
    assert flagged_claim.primary_anchor == "p.40:1-40:2"
    assert [assessment.anchor for assessment in flagged_claim.support_assessments] == [
        "p.40:1-40:2",
        "p.40:3-40:4",
    ]
    assert [assessment.role for assessment in flagged_claim.support_assessments] == [
        "contextual",
        "contextual",
    ]


class DelayedStubVerificationService:
    def __init__(self, outcomes: dict[uuid.UUID, VerificationExecution], delays: dict[uuid.UUID, float]) -> None:
        self.outcomes = outcomes
        self.delays = delays
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def verify_claim(
        self,
        claim_id: uuid.UUID,
        *,
        model_version: str | None = None,
        prompt_version: str | None = None,
    ) -> VerificationExecution:
        del model_version, prompt_version
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            time.sleep(self.delays[claim_id])
            return self.outcomes[claim_id]
        finally:
            with self._lock:
                self._active -= 1


def test_compile_draft_concurrency_preserves_order_and_structured_output(session: Session) -> None:
    _, source_document, draft = _build_workspace(session)
    first_claim = _add_claim(
        session,
        draft=draft,
        source_document=source_document,
        paragraph_index=1,
        claim_text="Claim one.",
        support_rows=[],
    )
    second_claim = _add_claim(
        session,
        draft=draft,
        source_document=source_document,
        paragraph_index=2,
        claim_text="Claim two.",
        support_rows=[],
    )
    third_claim = _add_claim(
        session,
        draft=draft,
        source_document=source_document,
        paragraph_index=3,
        claim_text="Claim three.",
        support_rows=[],
    )
    fourth_claim = _add_claim(
        session,
        draft=draft,
        source_document=source_document,
        paragraph_index=4,
        claim_text="Claim four.",
        support_rows=[],
    )
    session.commit()

    stub_service = DelayedStubVerificationService(
        outcomes={
            first_claim.id: _build_execution(
                first_claim.id,
                verdict=SupportStatus.UNSUPPORTED,
                flags=["subject_mismatch"],
                confidence=0.25,
            ),
            second_claim.id: _build_execution(
                second_claim.id,
                verdict=SupportStatus.SUPPORTED,
                flags=[],
                confidence=0.8,
            ),
            third_claim.id: _build_execution(
                third_claim.id,
                verdict=SupportStatus.AMBIGUOUS,
                flags=["contextual_support_only"],
                primary_anchor="p.50:1-50:2",
                assessments=[
                    ProviderSupportAssessment(
                        segment_id=uuid.uuid4(),
                        anchor="p.50:1-50:2",
                        role="contextual",
                        contribution="Question framing overlaps the claim.",
                    ),
                    ProviderSupportAssessment(
                        segment_id=uuid.uuid4(),
                        anchor="p.50:3-50:4",
                        role="contextual",
                        contribution="Equivocal answer does not state the proposition directly.",
                    ),
                ],
                confidence=0.4,
            ),
            fourth_claim.id: _build_execution(
                fourth_claim.id,
                verdict=SupportStatus.UNVERIFIED,
                flags=["missing_citation"],
                confidence=0.15,
            ),
        },
        delays={
            first_claim.id: 0.05,
            second_claim.id: 0.04,
            third_claim.id: 0.03,
            fourth_claim.id: 0.01,
        },
    )

    result = DraftCompileService(session, verification_service=stub_service).compile_draft(
        draft.id,
        max_concurrency=3,
    )

    assert stub_service.max_active > 1
    assert result.total_claims == 4
    assert result.counts.supported == 1
    assert result.counts.ambiguous == 1
    assert result.counts.unsupported == 1
    assert result.counts.unverified == 1
    assert [claim.claim_text for claim in result.flagged_claims] == [
        "Claim one.",
        "Claim three.",
        "Claim four.",
    ]
    assert [claim.verdict for claim in result.flagged_claims] == [
        SupportStatus.UNSUPPORTED,
        SupportStatus.AMBIGUOUS,
        SupportStatus.UNVERIFIED,
    ]
    assert result.flagged_claims[1].primary_anchor == "p.50:1-50:2"
    assert [assessment.anchor for assessment in result.flagged_claims[1].support_assessments] == [
        "p.50:1-50:2",
        "p.50:3-50:4",
    ]
    assert [assessment.role for assessment in result.flagged_claims[1].support_assessments] == [
        "contextual",
        "contextual",
    ]


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


def _seed_mixed_outcome_draft(session: Session) -> uuid.UUID:
    matter, source_document, draft = _build_workspace(session)

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
        claim_text="Doe signed the contract.",
        support_rows=[
            {
                "raw_text": "Q. Did Doe sign the contract?",
                "page_start": 30,
                "line_start": 1,
                "page_end": 30,
                "line_end": 2,
                "speaker": "Q",
                "segment_type": "QUESTION_BLOCK",
                "sequence_order": 1,
            }
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
                "page_start": 35,
                "line_start": 1,
                "page_end": 35,
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


def _seed_provider_backed_draft(session: Session) -> uuid.UUID:
    _, source_document, draft = _build_workspace(session)

    _add_claim(
        session,
        draft=draft,
        source_document=source_document,
        paragraph_index=1,
        claim_text="Smith signed the contract.",
        support_rows=[
            {
                "raw_text": "A. Smith signed the contract on March 1.",
                "page_start": 39,
                "line_start": 1,
                "page_end": 39,
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
        claim_text="Doe signed the contract on March 1.",
        support_rows=[
            {
                "raw_text": "Q. Did Doe sign the contract on March 1?",
                "page_start": 40,
                "line_start": 1,
                "page_end": 40,
                "line_end": 2,
                "speaker": "Q",
                "segment_type": "QUESTION_BLOCK",
                "sequence_order": 1,
            },
            {
                "raw_text": "A. I do not remember whether Doe signed it.",
                "page_start": 40,
                "line_start": 3,
                "page_end": 40,
                "line_end": 4,
                "speaker": "A",
                "segment_type": "ANSWER_BLOCK",
                "sequence_order": 2,
            },
        ],
    )

    session.commit()
    return draft.id


def _build_workspace(session: Session) -> tuple[Matter, SourceDocument, Draft]:
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
    return matter, source_document, draft


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


def _build_execution(
    claim_id: uuid.UUID,
    *,
    verdict: SupportStatus,
    flags: list[str],
    confidence: float | None,
    primary_anchor: str | None = None,
    assessments: list[ProviderSupportAssessment] | None = None,
) -> VerificationExecution:
    return VerificationExecution(
        run=VerificationRun(
            claim_unit_id=claim_id,
            model_version="stub-v1",
            prompt_version="prompt-v1",
            deterministic_flags=list(flags),
            verdict=verdict,
            reasoning="Stub reasoning.",
            suggested_fix=None,
            confidence_score=confidence,
        ),
        result=VerificationResult(
            model_version="stub-v1",
            prompt_version="prompt-v1",
            deterministic_flags=list(flags),
            verdict=verdict,
            reasoning="Stub reasoning.",
            suggested_fix=None,
            confidence_score=confidence,
            primary_anchor=primary_anchor,
            support_assessments=assessments or [],
        ),
    )
