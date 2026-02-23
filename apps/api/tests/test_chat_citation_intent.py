from __future__ import annotations

import json
from uuid import uuid4

from sqlmodel import SQLModel, Session, create_engine
from unittest.mock import Mock, patch

from app.db import CitationVerification, Document
from app.main import (
    CitationVerificationFinding,
    CitationVerificationSummary,
    ChatRequest,
    VerifyCitationsResponse,
    _citation_list_hash,
    chat,
)


def _make_session(tmp_path):
    db_path = tmp_path / "chat-intent-test.db"
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def _insert_document(session: Session) -> str:
    doc_id = uuid4()
    session.add(
        Document(
            doc_id=doc_id,
            filename="test.pdf",
            file_path="/tmp/test.pdf",
            stub_text="sample",
        )
    )
    session.commit()
    return str(doc_id)


def test_chat_citation_intent_calls_verification_helper(tmp_path) -> None:
    with _make_session(tmp_path) as session:
        doc_id = _insert_document(session)
        rows = [
            {
                "chunk_id": "chunk-a",
                "text": "Roe v. Wade, 410 U.S. 113.",
                "page": 1,
                "score": 0.9,
            }
        ]
        verification = VerifyCitationsResponse(
            findings=[
                CitationVerificationFinding(
                    citation="410 U.S. 113",
                    status="verified",
                    confidence=1.0,
                    best_match=None,
                    explanation="One CourtListener match was returned with an exact citation match.",
                    evidence="Roe v. Wade, 410 U.S. 113.",
                )
            ],
            summary=CitationVerificationSummary(
                total=1, verified=1, not_found=0, ambiguous=0
            ),
            citations=["410 U.S. 113"],
        )
        mock_verify = Mock(return_value=verification)

        with patch("app.main.query_document_chunks", return_value=rows):
            with patch(
                "app.main._run_citation_verification_for_citations", mock_verify
            ):
                response = chat(
                    ChatRequest(doc_id=doc_id, message="Please verify citations"),
                    session=session,
                )

        assert mock_verify.call_count == 1
        assert mock_verify.call_args.kwargs["citations"] == ["410 U.S. 113"]
        assert response.tool_result is not None
        assert response.tool_result.type == "citation_verification"
        assert response.tool_result.findings[0].citation == "410 U.S. 113"
        assert response.tool_result.findings[0].evidence == "Roe v. Wade, 410 U.S. 113."
        assert response.tool_result.summary.total == 1
        assert response.tool_result.citations == ["410 U.S. 113"]


def test_chat_citation_intent_uses_cache_on_hit(tmp_path) -> None:
    with _make_session(tmp_path) as session:
        doc_id = _insert_document(session)
        rows = [
            {
                "chunk_id": "chunk-cache",
                "doc_id": doc_id,
                "text": "Brown v. Board of Education, 347 U.S. 483.",
                "page": 2,
                "score": 0.8,
            }
        ]
        input_hash = _citation_list_hash(["347 U.S. 483"])
        raw_payload = {
            "results": [{"citation": "347 U.S. 483", "results": [{"id": 2}]}]
        }
        session.add(
            CitationVerification(
                input_hash=input_hash,
                doc_id=doc_id,
                chunk_id="chunk-cache",
                raw_json=json.dumps(raw_payload, sort_keys=True),
                summary_status="verified",
            )
        )
        session.commit()

        with patch("app.main.query_document_chunks", return_value=rows):
            with patch(
                "app.main.CourtListenerClient.lookup_citation_list",
                side_effect=AssertionError("lookup should not be called on cache hit"),
            ):
                response = chat(
                    ChatRequest(doc_id=doc_id, message="check citations"),
                    session=session,
                )

        assert response.tool_result is not None
        assert response.tool_result.findings[0].citation == "347 U.S. 483"
        assert "347 U.S. 483" in response.tool_result.findings[0].evidence
        assert response.tool_result.summary.verified == 1
        assert response.tool_result.citations == ["347 U.S. 483"]


def test_chat_citation_intent_no_citations_detected(tmp_path) -> None:
    with _make_session(tmp_path) as session:
        doc_id = _insert_document(session)
        rows = [
            {
                "chunk_id": "chunk-none",
                "text": "This chunk has no legal citation text.",
                "page": 1,
                "score": 0.7,
            }
        ]

        with patch("app.main.query_document_chunks", return_value=rows):
            with patch(
                "app.main._run_citation_verification_for_citations",
                side_effect=AssertionError("should not verify when no citations found"),
            ):
                response = chat(
                    ChatRequest(doc_id=doc_id, message="verify citations"),
                    session=session,
                )

        assert response.tool_result is not None
        assert response.tool_result.findings == []
        assert response.tool_result.summary.total == 0
        assert response.tool_result.citations == []


def test_chat_non_trigger_behavior_unchanged(tmp_path) -> None:
    with _make_session(tmp_path) as session:
        doc_id = _insert_document(session)
        rows = [
            {
                "chunk_id": "chunk-summary",
                "text": "This is an ordinary summary chunk without citation verification intent.",
                "page": 1,
                "score": 0.5,
            }
        ]

        with patch("app.main.query_document_chunks", return_value=rows):
            with patch(
                "app.main._run_citation_verification",
                side_effect=AssertionError(
                    "verification helper should not run for non-trigger message"
                ),
            ):
                response = chat(
                    ChatRequest(doc_id=doc_id, message="Summarize this document"),
                    session=session,
                )

        assert response.tool_result is None
        assert response.findings == []
        assert len(response.sources) == 1
        assert isinstance(response.answer, str)
        assert response.answer
