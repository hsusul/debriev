from __future__ import annotations

from datetime import UTC, datetime

from .citeextract.extract import extract_case_citations
from .types import CitationSpan, DebrievReport, VerificationResult


def extract_citations(text: str) -> list[CitationSpan]:
    return extract_case_citations(text)


def verify_citation(span: CitationSpan) -> VerificationResult:
    return VerificationResult(status="unverified", details={"raw": span.raw})


def score_document(text: str) -> DebrievReport:
    _ = text
    return DebrievReport(
        version="v1",
        overall_score=0,
        summary="Stub report",
        citations=[],
        created_at=datetime.now(UTC).isoformat(),
    )
