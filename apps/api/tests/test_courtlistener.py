from __future__ import annotations

from datetime import UTC, datetime, timedelta
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from typing import Any
import types
from unittest.mock import Mock, call, patch
from urllib.error import HTTPError
from uuid import UUID, uuid4

from fastapi import BackgroundTasks, HTTPException
from sqlmodel import SQLModel, Session, create_engine, select

from app.courtlistener import CourtListenerClient, CourtListenerError
from app.db import (
    CitationVerification,
    Document,
    Report,
    VerificationJob,
    VerificationResult,
)
from app.main import (
    ChatFinding,
    VerificationJobCreateRequest,
    VerifyCitationsRequest,
    VerifyExtractedCitationsRequest,
    _extract_case_name_near,
    _extract_case_name_by_citation,
    _extract_citations,
    _extract_citation_evidence,
    _extract_citation_evidence_from_chunks,
    _extract_citations_from_chunks,
    _reap_stale_verification_jobs,
    _run_citation_verification_for_citations,
    _courtlistener_raw_to_findings,
    _execute_verification_job,
    create_verification_job,
    export_report_pdf,
    get_report_citations,
    get_report_risk,
    get_report_verification_history,
    get_verification_job_status,
    get_report_verification,
    verify_citations,
    verify_citations_extracted,
)


class _MockHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> _MockHTTPResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


def _make_session(tmp_path: Path) -> Session:
    db_path = tmp_path / "courtlistener-test.db"
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def _http_error(status: int, body: str) -> HTTPError:
    return HTTPError(
        url="https://www.courtlistener.com/api/rest/v4/citation-lookup/",
        code=status,
        msg="error",
        hdrs=None,
        fp=BytesIO(body.encode("utf-8")),
    )


def _insert_document_with_report(session: Session) -> str:
    doc_id = uuid4()
    session.add(
        Document(
            doc_id=doc_id,
            filename="fixture.pdf",
            file_path="/tmp/fixture.pdf",
            stub_text="Roe v. Wade, 410 U.S. 113 (1973).",
        )
    )
    session.add(
        Report(
            doc_id=doc_id,
            report_json={
                "version": "v1",
                "overall_score": 0,
                "summary": "seed",
                "citations": [
                    {
                        "raw": "Roe v. Wade, 410 U.S. 113",
                        "context_text": "Roe v. Wade, 410 U.S. 113 (1973).",
                    }
                ],
            },
        )
    )
    session.commit()
    return str(doc_id)


def test_lookup_citations_missing_token_raises() -> None:
    with patch.dict(os.environ, {}, clear=True):
        with patch("app.settings.settings.courtlistener_token", None):
            client = CourtListenerClient(token_env="COURTLISTENER_TOKEN")
            try:
                client.lookup_citations("410 U.S. 113")
            except CourtListenerError as exc:
                assert "COURTLISTENER_TOKEN" in str(exc)
            else:
                raise AssertionError(
                    "Expected CourtListenerError when token is missing"
                )


def test_lookup_citations_sends_token_header() -> None:
    captured: dict[str, Any] = {}

    def _fake_urlopen(request: Any, timeout: float) -> _MockHTTPResponse:
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _MockHTTPResponse({"results": []})

    with patch.dict(os.environ, {"COURTLISTENER_TOKEN": "token-123"}, clear=False):
        with patch("app.courtlistener.urlopen", side_effect=_fake_urlopen):
            client = CourtListenerClient()
            response = client.lookup_citations("  Roe v. Wade  ")

    assert response == {"results": []}
    assert captured["authorization"] == "Token token-123"
    assert captured["payload"] == {"text": "Roe v. Wade"}
    assert captured["timeout"] == 10.0


def test_lookup_citations_retries_on_429_then_succeeds() -> None:
    calls = {"count": 0}

    def _flaky_urlopen(request: Any, timeout: float) -> _MockHTTPResponse:
        calls["count"] += 1
        if calls["count"] < 3:
            raise _http_error(429, "rate limited")
        return _MockHTTPResponse({"results": []})

    with patch.dict(os.environ, {"COURTLISTENER_TOKEN": "token-123"}, clear=False):
        with patch("app.courtlistener.urlopen", side_effect=_flaky_urlopen):
            with patch("app.courtlistener.time.sleep") as sleep_mock:
                client = CourtListenerClient(backoff_seconds=0.1, max_retries=3)
                response = client.lookup_citations("410 U.S. 113")

    assert response == {"results": []}
    assert calls["count"] == 3
    assert sleep_mock.call_args_list == [call(0.1), call(0.2)]


def test_lookup_citations_retries_on_5xx_then_raises() -> None:
    def _always_fails(request: Any, timeout: float) -> _MockHTTPResponse:
        raise _http_error(503, "temporary unavailable")

    with patch.dict(os.environ, {"COURTLISTENER_TOKEN": "token-123"}, clear=False):
        with patch("app.courtlistener.urlopen", side_effect=_always_fails):
            with patch("app.courtlistener.time.sleep") as sleep_mock:
                client = CourtListenerClient(backoff_seconds=0.1, max_retries=2)
                try:
                    client.lookup_citations("410 U.S. 113")
                except CourtListenerError as exc:
                    assert "status 503" in str(exc)
                else:
                    raise AssertionError(
                        "Expected CourtListenerError after retry attempts"
                    )

    assert sleep_mock.call_args_list == [call(0.1), call(0.2)]


def test_verify_citations_is_deterministic_and_sorted(tmp_path: Path) -> None:
    mock_payload = {
        "results": [
            {"citation": "410 U.S. 113", "results": [{"id": 2}, {"id": 1}]},
            {"citation": "347 U.S. 483", "results": [{"id": 3}]},
            {"citation": "410 U.S. 113", "results": [{"id": 1}, {"id": 4}]},
            {"citation": "123 F.3d 456", "results": []},
        ]
    }

    def _fake_urlopen(request: Any, timeout: float) -> _MockHTTPResponse:
        return _MockHTTPResponse(mock_payload)

    payload = VerifyCitationsRequest(text="Verify these citations")
    with patch.dict(os.environ, {"COURTLISTENER_TOKEN": "token-123"}, clear=False):
        with _make_session(tmp_path) as session:
            with patch("app.courtlistener.urlopen", side_effect=_fake_urlopen):
                first = verify_citations(payload, session=session).model_dump()
            with patch("app.courtlistener.urlopen", side_effect=_fake_urlopen):
                second = verify_citations(payload, session=session).model_dump()

    assert first == second
    assert [item["citation"] for item in first["findings"]] == [
        "123 F.3d 456",
        "347 U.S. 483",
        "410 U.S. 113",
    ]
    assert [item["status"] for item in first["findings"]] == [
        "not_found",
        "verified",
        "ambiguous",
    ]
    assert first["summary"] == {
        "total": 3,
        "verified": 1,
        "not_found": 1,
        "ambiguous": 1,
    }
    assert first["citations"] == ["123 F.3d 456", "347 U.S. 483", "410 U.S. 113"]
    assert first["findings"][0]["confidence"] == 0.0
    assert first["findings"][1]["confidence"] == 1.0
    assert first["findings"][2]["confidence"] == 0.5
    assert first["raw"] is None

    with patch.dict(os.environ, {"COURTLISTENER_TOKEN": "token-123"}, clear=False):
        with _make_session(tmp_path) as session:
            with patch("app.courtlistener.urlopen", side_effect=_fake_urlopen):
                with_raw = verify_citations(
                    VerifyCitationsRequest(
                        text="Verify these citations", include_raw=True
                    ),
                    session=session,
                ).model_dump()
    assert with_raw["raw"] == mock_payload


def test_verify_citations_missing_token_returns_http_error(tmp_path: Path) -> None:
    with patch.dict(os.environ, {}, clear=True):
        with patch("app.settings.settings.courtlistener_token", None):
            with _make_session(tmp_path) as session:
                try:
                    verify_citations(
                        VerifyCitationsRequest(text="410 U.S. 113"), session=session
                    )
                except HTTPException as exc:
                    assert exc.status_code == 503
                    assert "COURTLISTENER_TOKEN" in str(exc.detail)
                else:
                    raise AssertionError("Expected HTTPException when token is missing")


def test_verify_citations_cache_miss_stores_record(tmp_path: Path) -> None:
    payload = VerifyCitationsRequest(
        text="  410 U.S. 113  ", doc_id="doc-1", chunk_id="chunk-1"
    )
    mock_raw = {"results": [{"citation": "410 U.S. 113", "results": [{"id": 9}]}]}
    mock_lookup = Mock(return_value=mock_raw)

    with _make_session(tmp_path) as session:
        with patch("app.main.CourtListenerClient.lookup_citations", mock_lookup):
            response = verify_citations(payload, session=session).model_dump()

        assert mock_lookup.call_count == 1
        assert response["findings"][0]["citation"] == "410 U.S. 113"
        assert response["findings"][0]["status"] == "verified"
        assert response["summary"]["verified"] == 1
        assert response["citations"] == ["410 U.S. 113"]

        rows = session.exec(select(CitationVerification)).all()
        assert len(rows) == 1
        assert rows[0].doc_id == "doc-1"
        assert rows[0].chunk_id == "chunk-1"
        assert rows[0].summary_status == "verified"
        assert rows[0].raw_json == json.dumps(mock_raw, sort_keys=True)


def test_verify_citations_cache_hit_skips_client(tmp_path: Path) -> None:
    payload = VerifyCitationsRequest(text="410   U.S.   113")
    normalized = "410 U.S. 113"
    input_hash = sha256(normalized.encode("utf-8")).hexdigest()
    cached_raw = {"results": [{"citation": "410 U.S. 113", "results": [{"id": 1}]}]}

    with _make_session(tmp_path) as session:
        session.add(
            CitationVerification(
                input_hash=input_hash,
                doc_id="doc-cache",
                chunk_id="chunk-cache",
                raw_json=json.dumps(cached_raw, sort_keys=True),
                summary_status="verified",
            )
        )
        session.commit()

        with patch(
            "app.main.CourtListenerClient.lookup_citations",
            side_effect=AssertionError("client should not be called on cache hit"),
        ):
            response = verify_citations(payload, session=session).model_dump()

    assert response["findings"][0]["citation"] == "410 U.S. 113"
    assert response["summary"]["total"] == 1


def test_verify_citations_repeated_calls_are_identical(tmp_path: Path) -> None:
    payload = VerifyCitationsRequest(text="Verify deterministic output")
    mock_raw = {
        "results": [
            {"citation": "410 U.S. 113", "results": [{"id": 2}, {"id": 1}]},
            {"citation": "347 U.S. 483", "results": [{"id": 3}]},
        ]
    }
    mock_lookup = Mock(return_value=mock_raw)

    with _make_session(tmp_path) as session:
        with patch("app.main.CourtListenerClient.lookup_citations", mock_lookup):
            first = verify_citations(payload, session=session).model_dump()
            second = verify_citations(payload, session=session).model_dump()

    assert mock_lookup.call_count == 1
    assert first == second


def test_courtlistener_raw_to_findings_not_found() -> None:
    findings = _courtlistener_raw_to_findings(
        {"results": [{"citation": "999 U.S. 999", "results": []}]}
    )
    assert len(findings) == 1
    assert findings[0].citation == "999 U.S. 999"
    assert findings[0].status == "not_found"
    assert findings[0].confidence == 0.0
    assert findings[0].best_match is None


def test_courtlistener_raw_to_findings_single_exact_match() -> None:
    findings = _courtlistener_raw_to_findings(
        {
            "results": [
                {
                    "citation": "410 U.S. 113",
                    "results": [
                        {
                            "citation": "410 U.S. 113",
                            "case_name": "Roe v. Wade",
                            "court": "U.S. Supreme Court",
                            "year": 1973,
                            "url": "https://example.test/roe",
                        }
                    ],
                }
            ]
        }
    )
    assert len(findings) == 1
    assert findings[0].status == "verified"
    assert findings[0].confidence == 1.0
    assert findings[0].best_match is not None
    assert findings[0].best_match.case_name == "Roe v. Wade"


def test_courtlistener_raw_to_findings_normalizes_relative_url() -> None:
    findings = _courtlistener_raw_to_findings(
        {
            "results": [
                {
                    "citation": "410 U.S. 113",
                    "results": [
                        {
                            "citation": "410 U.S. 113",
                            "case_name": "Roe v. Wade",
                            "url": "/opinion/108713/roe-v-wade/",
                        }
                    ],
                }
            ]
        }
    )
    assert findings[0].best_match is not None
    assert (
        findings[0].best_match.url
        == "https://www.courtlistener.com/opinion/108713/roe-v-wade/"
    )


def test_courtlistener_raw_to_findings_ambiguous_is_deterministic() -> None:
    raw = {
        "results": [
            {
                "citation": "12 F.3d 34",
                "results": [
                    {
                        "citation": "12 F.3d 34",
                        "case_name": "Zulu v. Echo",
                        "year": 2002,
                        "url": "https://example.test/zulu",
                    },
                    {
                        "citation": "12 F.3d 34",
                        "case_name": "Alpha v. Bravo",
                        "year": 2001,
                        "url": "https://example.test/alpha",
                    },
                ],
            }
        ]
    }
    first = _courtlistener_raw_to_findings(raw)
    second = _courtlistener_raw_to_findings(raw)

    assert len(first) == 1
    assert first == second
    assert first[0].status == "ambiguous"
    assert first[0].confidence == 0.5
    assert first[0].best_match is not None
    assert first[0].best_match.case_name == "Alpha v. Bravo"
    assert [candidate.case_name for candidate in first[0].candidates] == [
        "Alpha v. Bravo",
        "Zulu v. Echo",
    ]


def test_extract_citations_dedupes_and_orders_deterministically() -> None:
    text = """
    The opinion cites 410 U.S. 113, 123 F.3d 456, and in 2001 includes 12 F. Supp. 2d 345.
    It also repeats 410  U.S. 113 and includes 600 S. Ct. 12 (2024).
    """
    extracted = _extract_citations(text)
    assert extracted == [
        "12 F. Supp. 2d 345",
        "123 F.3d 456",
        "410 U.S. 113",
        "600 S. Ct. 12",
    ]


def test_extract_citations_from_chunks_uses_chunk_text() -> None:
    chunks = [
        {
            "doc_id": "doc-b",
            "chunk_id": "chunk-2",
            "text": "Mentions 600 S. Ct. 12 in 2024.",
        },
        {
            "doc_id": "doc-a",
            "chunk_id": "chunk-1",
            "text": "Contains 410 U.S. 113 and 123 F.3d 456.",
        },
    ]
    assert _extract_citations_from_chunks(chunks) == [
        "123 F.3d 456",
        "410 U.S. 113",
        "600 S. Ct. 12",
    ]


def test_extract_citations_rejects_non_citation_numeric_patterns() -> None:
    text = """
    Internal ledger: 12 345 and section 12 F 345 should not count.
    Random numbers 410 113 are not citations.
    """
    assert _extract_citations(text) == []


def test_extract_citations_requires_year_for_lower_confidence_reporters() -> None:
    text = """
    Without years: 12 F. Supp. 2d 345 and 600 S. Ct. 12.
    High-confidence citations without years: 410 U.S. 113 and 123 F.3d 456 and 456 F.2d 789.
    """
    assert _extract_citations(text) == [
        "123 F.3d 456",
        "410 U.S. 113",
        "456 F.2d 789",
    ]


def test_extract_citations_accepts_lower_confidence_when_year_is_near() -> None:
    text = """
    In 2003, the court cited 12 F. Supp. 2d 345.
    By 2024 it also referenced 600 S. Ct. 12.
    """
    assert _extract_citations(text) == ["12 F. Supp. 2d 345", "600 S. Ct. 12"]


def test_run_citation_verification_for_citations_batches_and_cache(
    tmp_path: Path,
) -> None:
    citations = [f"{idx} U.S. {100 + idx}" for idx in range(1, 31)]
    call_batches: list[list[str]] = []

    def _mock_lookup(_self: object, citation_batch: list[str]) -> dict[str, Any]:
        call_batches.append(list(citation_batch))
        return {
            "results": [
                {
                    "citation": citation,
                    "results": [
                        {"citation": citation, "case_name": f"Case {citation}"}
                    ],
                }
                for citation in citation_batch
            ]
        }

    with _make_session(tmp_path) as session:
        with patch("app.main.CourtListenerClient.lookup_citation_list", _mock_lookup):
            first = _run_citation_verification_for_citations(
                citations=citations,
                session=session,
                batch_size=25,
            ).model_dump()

        assert len(call_batches) == 2
        assert len(call_batches[0]) == 25
        assert len(call_batches[1]) == 5
        assert first["summary"]["total"] == 30
        assert first["summary"]["verified"] == 30

        with patch(
            "app.main.CourtListenerClient.lookup_citation_list",
            side_effect=AssertionError("cache hit should not call lookup"),
        ):
            second = _run_citation_verification_for_citations(
                citations=list(reversed(citations)),
                session=session,
                batch_size=25,
            ).model_dump()

    assert first == second


def test_verify_citations_extracted_endpoint_returns_summary(tmp_path: Path) -> None:
    payload = VerifyExtractedCitationsRequest(
        text="See Roe v. Wade, 410 U.S. 113 and 123 F.3d 456."
    )
    mock_lookup = Mock(
        return_value={
            "results": [
                {"citation": "410 U.S. 113", "results": [{"citation": "410 U.S. 113"}]},
                {"citation": "123 F.3d 456", "results": []},
            ]
        }
    )
    with _make_session(tmp_path) as session:
        with patch("app.main.CourtListenerClient.lookup_citation_list", mock_lookup):
            response = verify_citations_extracted(payload, session=session).model_dump()

    assert response["citations"] == ["123 F.3d 456", "410 U.S. 113"]
    assert response["summary"] == {
        "total": 2,
        "verified": 1,
        "not_found": 1,
        "ambiguous": 0,
    }
    findings_by_citation = {item["citation"]: item for item in response["findings"]}
    assert "410 U.S. 113" in findings_by_citation["410 U.S. 113"]["evidence"]
    assert findings_by_citation["410 U.S. 113"]["probable_case_name"] == "Roe v. Wade"
    assert findings_by_citation["123 F.3d 456"]["evidence"] != ""
    assert findings_by_citation["123 F.3d 456"]["probable_case_name"] == "Roe v. Wade"


def test_verify_citations_extracted_persists_verification_result(
    tmp_path: Path,
) -> None:
    doc_id = str(uuid4())
    payload = VerifyExtractedCitationsRequest(
        text="See Roe v. Wade, 410 U.S. 113.",
        doc_id=doc_id,
    )
    mock_lookup = Mock(
        return_value={
            "results": [
                {"citation": "410 U.S. 113", "results": [{"citation": "410 U.S. 113"}]}
            ]
        }
    )

    with _make_session(tmp_path) as session:
        with patch("app.main.CourtListenerClient.lookup_citation_list", mock_lookup):
            response = verify_citations_extracted(payload, session=session).model_dump()

        rows = session.exec(
            select(VerificationResult).where(VerificationResult.doc_id == doc_id)
        ).all()

    assert len(rows) == 1
    stored = rows[0]
    assert json.loads(stored.citations_json) == response["citations"]
    assert json.loads(stored.summary_json) == response["summary"]
    assert isinstance(json.loads(stored.findings_json), list)


def test_verification_history_returns_newest_first(tmp_path: Path) -> None:
    doc_id = str(uuid4())
    payload_first = VerifyExtractedCitationsRequest(
        text="See Roe v. Wade, 410 U.S. 113.",
        doc_id=doc_id,
    )
    payload_second = VerifyExtractedCitationsRequest(
        text="See Roe v. Wade, 410 U.S. 113 and 123 F.3d 456.",
        doc_id=doc_id,
    )
    mock_lookup = Mock(
        side_effect=[
            {"results": [{"citation": "410 U.S. 113", "results": []}]},
            {
                "results": [
                    {"citation": "410 U.S. 113", "results": []},
                    {"citation": "123 F.3d 456", "results": []},
                ]
            },
        ]
    )

    with _make_session(tmp_path) as session:
        with patch("app.main.CourtListenerClient.lookup_citation_list", mock_lookup):
            verify_citations_extracted(payload_first, session=session)
            verify_citations_extracted(payload_second, session=session)

        history = get_report_verification_history(doc_id=UUID(doc_id), session=session)
        latest = get_report_verification(doc_id=UUID(doc_id), session=session)

    assert len(history) == 2
    assert history[0].id > history[1].id
    assert history[0].citations_count == 2
    assert history[1].citations_count == 1
    assert latest.citations == ["123 F.3d 456", "410 U.S. 113"]


def test_get_report_verification_returns_latest_stored_result(tmp_path: Path) -> None:
    doc_id = uuid4()
    payload = VerifyExtractedCitationsRequest(
        text="See Roe v. Wade, 410 U.S. 113 and 123 F.3d 456.",
        doc_id=str(doc_id),
    )
    mock_lookup = Mock(
        return_value={
            "results": [
                {"citation": "410 U.S. 113", "results": [{"citation": "410 U.S. 113"}]},
                {"citation": "123 F.3d 456", "results": []},
            ]
        }
    )

    with _make_session(tmp_path) as session:
        with patch("app.main.CourtListenerClient.lookup_citation_list", mock_lookup):
            verify_citations_extracted(payload, session=session)
        result = get_report_verification(doc_id=doc_id, session=session).model_dump()

    assert result["citations"] == ["123 F.3d 456", "410 U.S. 113"]
    assert result["summary"] == {
        "total": 2,
        "verified": 1,
        "not_found": 1,
        "ambiguous": 0,
    }


def test_verification_job_creation_returns_queued(tmp_path: Path) -> None:
    with _make_session(tmp_path) as session:
        doc_id = _insert_document_with_report(session)
        response = create_verification_job(
            doc_id=UUID(doc_id),
            payload=VerificationJobCreateRequest(text="Roe v. Wade, 410 U.S. 113."),
            background_tasks=BackgroundTasks(),
            session=session,
        )
        stored_job = session.exec(
            select(VerificationJob).where(VerificationJob.id == response.job_id)
        ).first()

    assert response.status == "queued"
    assert stored_job is not None
    assert stored_job.status == "queued"


def test_verification_job_completes_and_returns_done_status(tmp_path: Path) -> None:
    with _make_session(tmp_path) as session:
        doc_id = _insert_document_with_report(session)
        create_response = create_verification_job(
            doc_id=UUID(doc_id),
            payload=VerificationJobCreateRequest(text="Roe v. Wade, 410 U.S. 113."),
            background_tasks=BackgroundTasks(),
            session=session,
        )

        with patch(
            "app.main.CourtListenerClient.lookup_citation_list",
            return_value={
                "results": [
                    {
                        "citation": "410 U.S. 113",
                        "results": [{"citation": "410 U.S. 113"}],
                    }
                ]
            },
        ):
            _execute_verification_job(
                create_response.job_id, "Roe v. Wade, 410 U.S. 113.", session
            )

        status = get_verification_job_status(
            doc_id=UUID(doc_id), job_id=create_response.job_id, session=session
        )

    assert status.status == "done"
    assert status.result_id is not None
    assert status.summary is not None
    assert status.summary.total == 1
    assert status.error_text is None


def test_verification_job_returns_failed_status_on_error(tmp_path: Path) -> None:
    with _make_session(tmp_path) as session:
        doc_id = _insert_document_with_report(session)
        create_response = create_verification_job(
            doc_id=UUID(doc_id),
            payload=VerificationJobCreateRequest(text="Roe v. Wade, 410 U.S. 113."),
            background_tasks=BackgroundTasks(),
            session=session,
        )

        with patch(
            "app.main.CourtListenerClient.lookup_citation_list",
            side_effect=CourtListenerError("forced failure"),
        ):
            _execute_verification_job(
                create_response.job_id, "Roe v. Wade, 410 U.S. 113.", session
            )

        status = get_verification_job_status(
            doc_id=UUID(doc_id), job_id=create_response.job_id, session=session
        )

    assert status.status == "failed"
    assert status.summary is None
    assert status.result_id is None
    assert status.error_text is not None
    assert "forced failure" in status.error_text


def test_extract_citation_evidence_adds_ellipses_deterministically() -> None:
    text = (
        "Start context "
        + ("x" * 200)
        + " includes 410 U.S. 113 in the middle "
        + ("y" * 200)
    )
    evidence = _extract_citation_evidence(text, ["410 U.S. 113"], window=30)
    snippet = evidence["410 U.S. 113"]
    assert snippet.startswith("…")
    assert snippet.endswith("…")
    assert "410 U.S. 113" in snippet


def test_extract_citation_evidence_from_chunks_uses_earliest_matching_chunk() -> None:
    chunks = [
        {"chunk_id": "chunk-1", "text": "No citation here."},
        {
            "chunk_id": "chunk-2",
            "text": "Relevant text includes 123 F.3d 456 and more.",
        },
        {"chunk_id": "chunk-3", "text": "Another mention 123 F.3d 456 later."},
    ]
    evidence = _extract_citation_evidence_from_chunks(chunks, ["123 F.3d 456"])
    assert evidence["123 F.3d 456"].startswith("Relevant text includes 123 F.3d 456")


def test_extract_citation_evidence_empty_when_not_found() -> None:
    text = "This has no reporter citation."
    evidence = _extract_citation_evidence(text, ["410 U.S. 113"])
    assert evidence["410 U.S. 113"] == ""


def test_extract_case_name_near_finds_preceding_case_name() -> None:
    text = "Roe v. Wade, 410 U.S. 113 (1973)."
    span_start = text.index("410 U.S. 113")
    span = (span_start, span_start + len("410 U.S. 113"))
    assert _extract_case_name_near(text, span) == "Roe v. Wade"


def test_extract_case_name_near_returns_none_without_case_pattern() -> None:
    text = "This sentence references 410 U.S. 113 with no nearby caption."
    span_start = text.index("410 U.S. 113")
    span = (span_start, span_start + len("410 U.S. 113"))
    assert _extract_case_name_near(text, span) is None


def test_extract_case_name_near_is_deterministic_with_multiple_candidates() -> None:
    text = (
        "Alpha v. Beta and Gamma v. Delta appear before the citation 410 U.S. 113. "
        "Gamma v. Delta is closer."
    )
    span_start = text.index("410 U.S. 113")
    span = (span_start, span_start + len("410 U.S. 113"))
    assert _extract_case_name_near(text, span) == "Gamma v. Delta"


def test_extract_case_name_by_citation_map_is_stable() -> None:
    text = "Roe v. Wade, 410 U.S. 113. Brown v. Board, 347 U.S. 483."
    mapping = _extract_case_name_by_citation(text, ["347 U.S. 483", "410 U.S. 113"])
    assert mapping == {
        "347 U.S. 483": "Brown v. Board",
        "410 U.S. 113": "Roe v. Wade",
    }


def test_get_report_citations_returns_persisted_extraction(tmp_path: Path) -> None:
    doc_id = str(uuid4())
    payload = VerifyExtractedCitationsRequest(
        text="Roe v. Wade, 410 U.S. 113 (1973). Brown v. Board, 347 U.S. 483 (1954).",
        doc_id=doc_id,
    )
    mock_lookup = Mock(
        return_value={
            "results": [
                {"citation": "347 U.S. 483", "results": [{"citation": "347 U.S. 483"}]},
                {"citation": "410 U.S. 113", "results": [{"citation": "410 U.S. 113"}]},
            ]
        }
    )

    with _make_session(tmp_path) as session:
        with patch("app.main.CourtListenerClient.lookup_citation_list", mock_lookup):
            verify_citations_extracted(payload, session=session)
        stored = get_report_citations(doc_id=UUID(doc_id), session=session).model_dump()

    assert stored["citations"] == ["347 U.S. 483", "410 U.S. 113"]
    assert list(stored["evidence"]) == ["347 U.S. 483", "410 U.S. 113"]
    assert stored["probable_case_name"] == {
        "347 U.S. 483": "Brown v. Board",
        "410 U.S. 113": "Roe v. Wade",
    }


def test_get_report_risk_is_deterministic_with_bogus_overlay(tmp_path: Path) -> None:
    doc_id = str(uuid4())
    payload = VerifyExtractedCitationsRequest(
        text="Roe v. Wade, 410 U.S. 113 (1973). Brown v. Board, 347 U.S. 483 (1954).",
        doc_id=doc_id,
    )
    mock_lookup = Mock(
        return_value={
            "results": [
                {"citation": "347 U.S. 483", "results": []},
                {"citation": "410 U.S. 113", "results": []},
            ]
        }
    )
    bogus_findings = [
        ChatFinding(
            case_name="Roe v. Wade",
            reason_label="nonexistent_case",
            reason_phrase="does not appear to exist",
            evidence="Roe v. Wade, 410 U.S. 113 does not appear to exist.",
            doc_id=doc_id,
            chunk_id="chunk-1",
        )
    ]

    with _make_session(tmp_path) as session:
        with patch("app.main.CourtListenerClient.lookup_citation_list", mock_lookup):
            verify_citations_extracted(payload, session=session)
        with patch("app.main._get_bogus_findings_for_doc", return_value=bogus_findings):
            first = get_report_risk(doc_id=UUID(doc_id), session=session).model_dump()
            second = get_report_risk(doc_id=UUID(doc_id), session=session).model_dump()

    assert first == second
    assert first["score"] == 85
    assert first["totals"] == {
        "verified": 0,
        "not_found": 2,
        "ambiguous": 0,
        "bogus": 1,
    }
    assert [item["citation"] for item in first["top_risks"]] == [
        "347 U.S. 483",
        "410 U.S. 113",
    ]


def test_reap_stale_verification_jobs_marks_old_jobs_failed(tmp_path: Path) -> None:
    with _make_session(tmp_path) as session:
        stale = VerificationJob(
            id="job-stale",
            doc_id=str(uuid4()),
            status="queued",
            created_at=datetime.now(UTC) - timedelta(hours=1),
            updated_at=datetime.now(UTC) - timedelta(hours=1),
        )
        fresh = VerificationJob(
            id="job-fresh",
            doc_id=str(uuid4()),
            status="running",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        session.add(stale)
        session.add(fresh)
        session.commit()

        updated = _reap_stale_verification_jobs(session, stale_minutes=15)
        stale_after = session.exec(
            select(VerificationJob).where(VerificationJob.id == stale.id)
        ).first()
        fresh_after = session.exec(
            select(VerificationJob).where(VerificationJob.id == fresh.id)
        ).first()

    assert updated == 1
    assert stale_after is not None
    assert stale_after.status == "failed"
    assert stale_after.error_text is not None
    assert fresh_after is not None
    assert fresh_after.status == "running"


def test_execute_verification_job_is_idempotent_when_result_present(
    tmp_path: Path,
) -> None:
    with _make_session(tmp_path) as session:
        doc_id = str(uuid4())
        job = VerificationJob(
            id="job-idempotent",
            doc_id=doc_id,
            status="running",
            result_id=123,
        )
        session.add(job)
        session.commit()

        with patch(
            "app.main._run_extracted_verification",
            side_effect=AssertionError("should not rerun completed job"),
        ):
            _execute_verification_job(job_id=job.id, text=None, session=session)

        stored = session.exec(
            select(VerificationJob).where(VerificationJob.id == job.id)
        ).first()

    assert stored is not None
    assert stored.status == "done"
    assert stored.result_id == 123


def test_export_report_pdf_returns_pdf_and_is_deterministic(
    tmp_path: Path, monkeypatch: Any
) -> None:
    class _FakeCanvas:
        def __init__(
            self, buffer: BytesIO, pagesize: tuple[float, float], invariant: int = 1
        ) -> None:
            self._buffer = buffer
            self._lines: list[str] = []
            self._pagesize = pagesize
            self._invariant = invariant

        def setFont(self, name: str, size: int) -> None:
            self._lines.append(f"FONT:{name}:{size}")

        def drawString(self, x: float, y: float, text: str) -> None:
            self._lines.append(f"TEXT:{x:.1f}:{y:.1f}:{text}")

        def showPage(self) -> None:
            self._lines.append("PAGE")

        def save(self) -> None:
            payload = "\n".join(
                ["%PDF-1.4", f"SIZE:{self._pagesize}", f"INV:{self._invariant}"]
                + self._lines
            )
            self._buffer.write(payload.encode("utf-8"))

    fake_reportlab = types.ModuleType("reportlab")
    fake_reportlab_lib = types.ModuleType("reportlab.lib")
    fake_pagesizes = types.ModuleType("reportlab.lib.pagesizes")
    fake_pagesizes.LETTER = (612.0, 792.0)
    fake_pdfgen = types.ModuleType("reportlab.pdfgen")
    fake_canvas = types.ModuleType("reportlab.pdfgen.canvas")
    fake_canvas.Canvas = _FakeCanvas

    monkeypatch.setitem(sys.modules, "reportlab", fake_reportlab)
    monkeypatch.setitem(sys.modules, "reportlab.lib", fake_reportlab_lib)
    monkeypatch.setitem(sys.modules, "reportlab.lib.pagesizes", fake_pagesizes)
    monkeypatch.setitem(sys.modules, "reportlab.pdfgen", fake_pdfgen)
    monkeypatch.setitem(sys.modules, "reportlab.pdfgen.canvas", fake_canvas)

    with _make_session(tmp_path) as session:
        doc_id = uuid4()
        session.add(
            Document(
                doc_id=doc_id,
                filename="fixture.pdf",
                file_path="/tmp/fixture.pdf",
                stub_text="stub",
            )
        )
        session.add(
            VerificationResult(
                doc_id=str(doc_id),
                input_hash="input",
                citations_hash="citations",
                citations_json=json.dumps(["410 U.S. 113"]),
                findings_json=json.dumps(
                    [
                        {
                            "citation": "410 U.S. 113",
                            "status": "not_found",
                            "confidence": 0.0,
                            "best_match": None,
                            "candidates": [],
                            "explanation": "No CourtListener matches were returned for this citation.",
                            "evidence": "Roe v. Wade, 410 U.S. 113",
                            "probable_case_name": "Roe v. Wade",
                        }
                    ]
                ),
                summary_json=json.dumps(
                    {"total": 1, "verified": 0, "not_found": 1, "ambiguous": 0}
                ),
            )
        )
        session.commit()

        with patch("app.main._get_bogus_findings_for_doc", return_value=[]):
            first = export_report_pdf(doc_id=doc_id, session=session)
            second = export_report_pdf(doc_id=doc_id, session=session)

    assert first.media_type == "application/pdf"
    assert second.media_type == "application/pdf"
    assert first.body == second.body
    assert first.body.startswith(b"%PDF-1.4")
