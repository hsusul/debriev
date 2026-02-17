from __future__ import annotations

from debriev_core.types import CitationSpan

from .regex_rules import CASE_CITATION_PATTERNS


def _make_context(text: str, start: int, end: int, radius: int = 60) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return text[left:right].strip()


def extract_case_citations(text: str) -> list[CitationSpan]:
    seen: set[tuple[int, int, str]] = set()
    spans: list[CitationSpan] = []

    for pattern in CASE_CITATION_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group(0)
            start, end = match.span()
            key = (start, end, raw)
            if key in seen:
                continue
            seen.add(key)

            spans.append(
                CitationSpan(
                    raw=raw,
                    normalized=" ".join(raw.split()),
                    start=start,
                    end=end,
                    context_text=_make_context(text, start, end),
                )
            )

    spans.sort(key=lambda s: (s.start if s.start is not None else -1, s.raw))
    return spans
