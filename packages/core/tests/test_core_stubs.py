from debriev_core.stubs import extract_citations, score_document, verify_citation
from debriev_core.types import CitationSpan, DebrievReport, VerificationResult


def test_contract_models_construct() -> None:
    span = CitationSpan(raw="410 U.S. 113")
    verification = VerificationResult(status="unverified")
    report = DebrievReport(
        version="v1",
        overall_score=0,
        summary="Stub report",
        citations=[{"raw": "410 U.S. 113", "verification_status": "unverified"}],
        created_at="2026-02-17T00:00:00Z",
    )

    assert span.raw == "410 U.S. 113"
    assert verification.status == "unverified"
    assert report.version == "v1"


def test_stub_functions() -> None:
    spans = extract_citations("Roe v. Wade, 410 U.S. 113")
    verification = verify_citation(CitationSpan(raw="410 U.S. 113"))
    report = score_document("stub text")

    assert len(spans) == 1
    assert spans[0].raw == "410 U.S. 113"
    assert verification.status == "unverified"
    assert verification.details == {"raw": "410 U.S. 113"}
    assert report.version == "v1"
    assert report.summary == "Stub report"
