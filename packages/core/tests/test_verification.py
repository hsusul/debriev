from __future__ import annotations

from typing import Any

from debriev_core.verify.courtlistener import verify_case_citation
from debriev_core.verify.parse import parse_case_citation


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError("http error")

    def json(self) -> dict[str, Any]:
        return self._payload


def test_parse_case_citation_us() -> None:
    parsed = parse_case_citation("410 U.S. 113")

    assert parsed is not None
    assert parsed.volume == 410
    assert parsed.reporter == "US"
    assert parsed.page == 113


def test_parse_case_citation_rejects_statute() -> None:
    assert parse_case_citation("18 U.S.C. 1030") is None


def test_verify_case_citation_unsupported_reporter() -> None:
    result = verify_case_citation("123 F.3d 456")

    assert result.status == "unverified"
    assert result.details == {"reason": "unsupported_reporter_v1"}


def test_verify_case_citation_verified_with_stubbed_http() -> None:
    def fake_get(*args: Any, **kwargs: Any) -> _FakeResponse:
        _ = (args, kwargs)
        return _FakeResponse(
            {
                "count": 1,
                "results": [
                    {
                        "caseName": "Roe v. Wade",
                        "absolute_url": "/opinion/108713/roe-v-wade/",
                    }
                ],
            }
        )

    result = verify_case_citation("410 U.S. 113", http_get=fake_get)

    assert result.status == "verified"
    assert result.details is not None
    assert result.details["courtlistener_url"] == "https://www.courtlistener.com/opinion/108713/roe-v-wade/"
