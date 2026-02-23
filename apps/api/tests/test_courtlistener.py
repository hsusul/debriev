from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

from fastapi import HTTPException
from sqlmodel import SQLModel, Session, create_engine, select

from app.courtlistener import CourtListenerClient, CourtListenerError
from app.db import CitationVerification
from app.main import (
    VerifyCitationsRequest,
    _courtlistener_raw_to_findings,
    verify_citations,
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


def test_lookup_citations_missing_token_raises() -> None:
    with patch.dict(os.environ, {}, clear=True):
        client = CourtListenerClient(token_env="COURTLISTENER_TOKEN")
        try:
            client.lookup_citations("410 U.S. 113")
        except CourtListenerError as exc:
            assert "COURTLISTENER_TOKEN" in str(exc)
        else:
            raise AssertionError("Expected CourtListenerError when token is missing")


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
