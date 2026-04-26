"""Source-agnostic evidence-role helpers for verification."""

from typing import Literal
import re


EvidenceRole = Literal["substantive", "contextual", "neutral"]

_DECLARATION_CONTEXTUAL_PATTERNS = (
    re.compile(r"\bdeclare as follows\b"),
    re.compile(r"\bunder penalty of perjury\b"),
    re.compile(r"\bforegoing is true\b"),
    re.compile(r"\bforegoing is true and correct\b"),
    re.compile(r"\bexecuted on\b"),
    re.compile(r"\bexecuted this\b"),
)
_EXHIBIT_CONTEXTUAL_PATTERNS = (
    re.compile(r"\btrue and correct copy\b"),
    re.compile(r"\battached hereto\b"),
    re.compile(r"\bpage intentionally left blank\b"),
    re.compile(r"\bmarked as exhibit\b"),
    re.compile(r"^exhibit\s+[a-z0-9._-]+$"),
)
_EXHIBIT_CONTEXTUAL_LABELS = {
    "from",
    "to",
    "cc",
    "bcc",
    "date",
    "sent",
    "subject",
    "re",
}
_WEAK_SUBSTANTIVE_PATTERNS = (
    re.compile(r"\bi\s+(?:do\s+not|don't|did\s+not|didn't)\s+(?:know|remember|recall)\b"),
    re.compile(r"\b(?:not sure|unsure)\b"),
    re.compile(r"\b(?:cannot|can't)\s+(?:say|recall|remember)\b"),
    re.compile(r"\b(?:no recollection|do not recall whether|don't know whether)\b"),
)


def determine_evidence_role(
    *,
    segment_type: str,
    speaker: str | None,
    raw_text: str,
) -> EvidenceRole:
    """Map source-specific segment shapes to source-agnostic evidence roles."""

    normalized_text = _normalize_text(raw_text)

    if segment_type == "QUESTION_BLOCK" or speaker == "Q":
        return "contextual"
    if segment_type == "ANSWER_BLOCK" or speaker == "A":
        return "substantive"
    if segment_type == "DECLARATION_PARAGRAPH":
        return "substantive"
    if segment_type == "DECLARATION_BLOCK":
        if _is_contextual_declaration_block(normalized_text):
            return "contextual"
        return "substantive"
    if segment_type == "EXHIBIT_LABELED_BLOCK":
        if _is_contextual_exhibit_label(speaker) or _is_contextual_exhibit_text(normalized_text):
            return "contextual"
        return "substantive"
    if segment_type in {"EXHIBIT_PAGE_BLOCK", "EXHIBIT_TEXT_BLOCK"}:
        if _is_contextual_exhibit_text(normalized_text):
            return "contextual"
        return "substantive"
    if segment_type == "UNANCHORED_TEXT":
        return "contextual"
    return "neutral"


def is_substantive_evidence_role(role: EvidenceRole) -> bool:
    return role == "substantive"


def is_non_substantive_evidence_role(role: EvidenceRole) -> bool:
    return role != "substantive"


def is_weak_substantive_text(raw_text: str) -> bool:
    normalized_text = _normalize_text(raw_text)
    return any(pattern.search(normalized_text) for pattern in _WEAK_SUBSTANTIVE_PATTERNS)


def _is_contextual_declaration_block(normalized_text: str) -> bool:
    return any(pattern.search(normalized_text) for pattern in _DECLARATION_CONTEXTUAL_PATTERNS)


def _is_contextual_exhibit_label(label: str | None) -> bool:
    if label is None:
        return False
    return label.strip().lower() in _EXHIBIT_CONTEXTUAL_LABELS


def _is_contextual_exhibit_text(normalized_text: str) -> bool:
    return any(pattern.search(normalized_text) for pattern in _EXHIBIT_CONTEXTUAL_PATTERNS)


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())
