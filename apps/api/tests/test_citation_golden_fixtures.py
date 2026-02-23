from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from sqlmodel import SQLModel, Session, create_engine

from app.main import _extract_citations, _run_citation_verification_for_citations

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
_FIXTURE_EXPECTED: dict[str, list[str]] = {
    "fixture_roe.txt": ["410 U.S. 113"],
    "fixture_fake.txt": ["410 U.S. 113", "999 F.3d 999"],
    "fixture_noise.txt": [],
}


def _load_fixture(name: str) -> str:
    return (_FIXTURE_DIR / name).read_text(encoding="utf-8")


def _make_session(tmp_path: Path) -> Session:
    db_path = tmp_path / "citation-golden-test.db"
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def test_golden_fixture_extraction_matches_expected_exactly() -> None:
    for fixture_name, expected in _FIXTURE_EXPECTED.items():
        extracted = _extract_citations(_load_fixture(fixture_name))
        assert extracted == expected


def test_golden_fixture_ordering_is_stable() -> None:
    for fixture_name in _FIXTURE_EXPECTED:
        text = _load_fixture(fixture_name)
        first = _extract_citations(text)
        second = _extract_citations(text)
        assert first == second


def test_golden_fixture_precision_proxy_subset_present() -> None:
    total_expected = 0
    total_found = 0
    for fixture_name, expected in _FIXTURE_EXPECTED.items():
        extracted = _extract_citations(_load_fixture(fixture_name))
        total_expected += len(expected)
        total_found += sum(1 for citation in expected if citation in extracted)

    precision_proxy = total_found / total_expected if total_expected else 1.0
    assert precision_proxy == 1.0


def test_golden_fixture_mocked_verification_summary(tmp_path: Path) -> None:
    citations = _extract_citations(_load_fixture("fixture_fake.txt"))

    def _mock_lookup(_self: object, citation_batch: list[str]) -> dict[str, Any]:
        return {
            "results": [
                {
                    "citation": citation,
                    "results": (
                        [{"citation": citation, "case_name": "Roe v. Wade"}]
                        if citation == "410 U.S. 113"
                        else []
                    ),
                }
                for citation in citation_batch
            ]
        }

    with _make_session(tmp_path) as session:
        with patch("app.main.CourtListenerClient.lookup_citation_list", _mock_lookup):
            response = _run_citation_verification_for_citations(
                citations=citations,
                session=session,
            )

    assert response.citations == ["410 U.S. 113", "999 F.3d 999"]
    statuses = {finding.citation: finding.status for finding in response.findings}
    assert statuses["410 U.S. 113"] == "verified"
    assert statuses["999 F.3d 999"] == "not_found"
    assert response.summary.total == 2
    assert response.summary.verified == 1
    assert response.summary.not_found == 1
    assert response.summary.ambiguous == 0
