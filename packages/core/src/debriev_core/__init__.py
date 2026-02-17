from .stubs import extract_citations, score_document, verify_citation
from .types import CitationSpan, DebrievReport, VerificationResult
from .verify import parse_case_citation, verify_case_citation

__all__ = [
    "CitationSpan",
    "VerificationResult",
    "DebrievReport",
    "extract_citations",
    "verify_citation",
    "score_document",
    "parse_case_citation",
    "verify_case_citation",
]
