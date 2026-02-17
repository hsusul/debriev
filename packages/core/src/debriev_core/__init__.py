from .stubs import extract_citations, score_document, verify_citation
from .types import CitationSpan, DebrievReport, VerificationResult

__all__ = [
    "CitationSpan",
    "VerificationResult",
    "DebrievReport",
    "extract_citations",
    "verify_citation",
    "score_document",
]
