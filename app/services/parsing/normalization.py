"""Normalization helpers shared across parsing and matching."""

import re

WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Collapse whitespace while preserving casing."""

    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_for_match(text: str) -> str:
    """Normalize text for lightweight matching heuristics."""

    normalized = normalize_text(text)
    return normalized.lower()

