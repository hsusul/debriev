from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import patch

from fastapi import HTTPException

from app.courtlistener import CourtListenerClient, CourtListenerError
from app.main import VerifyCitationsRequest, verify_citations


class _MockHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> _MockHTTPResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


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


def test_verify_citations_is_deterministic_and_sorted() -> None:
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
        with patch("app.courtlistener.urlopen", side_effect=_fake_urlopen):
            first = verify_citations(payload).model_dump()
        with patch("app.courtlistener.urlopen", side_effect=_fake_urlopen):
            second = verify_citations(payload).model_dump()

    assert first == second
    assert [item["citation"] for item in first["verified"]] == [
        "123 F.3d 456",
        "347 U.S. 483",
        "410 U.S. 113",
    ]
    assert [item["id"] for item in first["verified"][2]["results"]] == [1, 2, 4]
    assert first["raw"] is None

    with patch.dict(os.environ, {"COURTLISTENER_TOKEN": "token-123"}, clear=False):
        with patch("app.courtlistener.urlopen", side_effect=_fake_urlopen):
            with_raw = verify_citations(
                VerifyCitationsRequest(text="Verify these citations", include_raw=True)
            ).model_dump()
    assert with_raw["raw"] == mock_payload


def test_verify_citations_missing_token_returns_http_error() -> None:
    with patch.dict(os.environ, {}, clear=True):
        try:
            verify_citations(VerifyCitationsRequest(text="410 U.S. 113"))
        except HTTPException as exc:
            assert exc.status_code == 503
            assert "COURTLISTENER_TOKEN" in str(exc.detail)
        else:
            raise AssertionError("Expected HTTPException when token is missing")
