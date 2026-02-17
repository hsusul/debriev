from __future__ import annotations

from typing import Any, Callable

import requests

from debriev_core.types import CitationSpan, VerificationResult

from .parse import parse_case_citation

COURTLISTENER_SEARCH_URL = "https://www.courtlistener.com/api/rest/v3/search/"
COURTLISTENER_BASE_URL = "https://www.courtlistener.com"


HttpGet = Callable[..., requests.Response]


def _to_absolute_url(value: str | None) -> str | None:
    if not value:
        return None
    if value.startswith("http://") or value.startswith("https://"):
        return value
    return f"{COURTLISTENER_BASE_URL}{value}"


def verify_case_citation(raw: str | CitationSpan, http_get: HttpGet = requests.get) -> VerificationResult:
    raw_text = raw.raw if isinstance(raw, CitationSpan) else raw

    parsed = parse_case_citation(raw_text)
    if parsed is None:
        return VerificationResult(status="unverified", details={"reason": "unparseable_citation"})

    if parsed.reporter != "US":
        return VerificationResult(status="unverified", details={"reason": "unsupported_reporter_v1"})

    params = {"type": "o", "q": raw_text}
    headers = {"User-Agent": "debriev/0.1 (+https://github.com/)"}

    try:
        response = http_get(COURTLISTENER_SEARCH_URL, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return VerificationResult(status="error", details={"reason": "request_error", "message": str(exc)})
    except ValueError as exc:
        return VerificationResult(status="error", details={"reason": "invalid_json", "message": str(exc)})

    count = int(payload.get("count", 0)) if isinstance(payload, dict) else 0
    results = payload.get("results", []) if isinstance(payload, dict) else []

    if count <= 0 or not isinstance(results, list) or not results:
        return VerificationResult(status="not_found", details={"reason": "no_match", "query": raw_text})

    first = results[0] if isinstance(results[0], dict) else {}
    absolute_url = _to_absolute_url(first.get("absolute_url"))
    case_name = first.get("caseName") or first.get("case_name")
    matched_citation = first.get("citation") or raw_text

    return VerificationResult(
        status="verified",
        details={
            "matched_citation": matched_citation,
            "case_name": case_name,
            "courtlistener_url": absolute_url,
        },
    )
