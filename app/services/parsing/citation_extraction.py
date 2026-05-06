"""Typed citation extraction backed by eyecite for the narrow MVP surface."""

from dataclasses import dataclass
import re
from typing import Any

from eyecite import get_citations
from eyecite.models import FullCaseCitation

from app.services.parsing.normalization import normalize_text

CASE_NAME_TOKEN = r"[A-Z][A-Za-z0-9'’.&-]*"
CASE_NAME_CONNECTOR = r"(?:of|and|the|for|in|on|at|to|by|ex rel\.)"
CASE_NAME_WORD = rf"(?:{CASE_NAME_TOKEN}|{CASE_NAME_CONNECTOR})"
CASE_NAME_SUFFIX = r"(?:,\s+(?:Inc\.|Corp\.|Co\.|Ltd\.|LLC))?"
CASE_NAME_PARTY = rf"{CASE_NAME_TOKEN}(?:\s+{CASE_NAME_WORD}){{0,7}}{CASE_NAME_SUFFIX}"
REPORTER_TOKEN = r"[A-Z][A-Za-z0-9.\-]*"
REPORTER_PART = rf"{REPORTER_TOKEN}(?:\s+{REPORTER_TOKEN})*"
FULL_CASE_CITATION_RE = re.compile(
    rf"\b(?P<case_name>{CASE_NAME_PARTY}\s+v\.\s+{CASE_NAME_PARTY})"
    rf",\s*(?P<volume>\d{{1,4}})\s+(?P<reporter>{REPORTER_PART})\s+(?P<page>\d{{1,5}})"
    rf"(?:,\s*(?P<pin_cite>\d{{1,5}}(?:-\d{{1,5}})?))?"
    rf"(?:\s*\((?P<parenthetical>[^()]*?\d{{4}})\))?"
)
PARENTHETICAL_YEAR_RE = re.compile(r"^(?:(?P<court>.*\S)\s+)?(?P<year>\d{4})$")


@dataclass(slots=True, frozen=True)
class CitationSpan:
    start: int
    end: int


@dataclass(slots=True, frozen=True)
class CitationCandidate:
    citation_text: str
    span: CitationSpan
    case_name: str | None
    volume: str | None
    reporter: str | None
    page: str | None
    pin_cite: str | None
    court: str | None
    year: int | None
    citation_kind: str
    parse_status: str
    normalized_resource_key: str | None

    @property
    def is_full_case_citation(self) -> bool:
        return (
            self.citation_kind == "full_case"
            and self.parse_status == "full_case_parsed"
            and self.case_name is not None
            and self.volume is not None
            and self.reporter is not None
            and self.page is not None
        )


class CitationExtractionService:
    """Compile raw text into typed citation candidates.

    Debriev's MVP exposes only full case citations. Other eyecite citation
    classes are intentionally classified internally and left out of the public
    citation-verification rows until antecedent resolution has explicit
    product semantics.
    """

    def extract(self, text: str) -> list[CitationCandidate]:
        if not text.strip():
            return []

        candidates: list[CitationCandidate] = []
        for citation in get_citations(text):
            candidate = self._candidate_from_eyecite_citation(text, citation)
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def extract_full_case_citations(self, text: str) -> list[CitationCandidate]:
        candidates = [candidate for candidate in self.extract(text) if candidate.is_full_case_citation]
        candidates.extend(_fallback_full_case_candidates(text, candidates))
        return sorted(_dedupe_candidates(candidates), key=lambda candidate: (candidate.span.start, candidate.span.end))

    def parse_full_case_citation(self, text: str) -> CitationCandidate | None:
        full_case_candidates = self.extract_full_case_citations(text)
        if not full_case_candidates:
            return None
        normalized_text = normalize_text(text).strip(" ,.;:")
        for candidate in full_case_candidates:
            if candidate.span.start == 0 and candidate.span.end == len(normalized_text):
                return candidate
        return full_case_candidates[0]

    def _candidate_from_eyecite_citation(self, text: str, citation: Any) -> CitationCandidate | None:
        citation_kind = _citation_kind(citation)
        start, end = _citation_span(citation)
        if start is None or end is None or start >= end:
            return None

        citation_text, start = _trim_leading_noncitation_lines(text[start:end], start)
        if not citation_text:
            return None

        if isinstance(citation, FullCaseCitation):
            metadata = citation.metadata
            groups = citation.groups
            case_name = _build_case_name(
                getattr(metadata, "plaintiff", None),
                getattr(metadata, "defendant", None),
            )
            volume = groups.get("volume")
            reporter = groups.get("reporter")
            page = groups.get("page")
            pin_cite = _normalize_pin_cite(getattr(metadata, "pin_cite", None))
            court = _normalize_court(getattr(metadata, "court", None))
            year = _parse_year(getattr(metadata, "year", None))
            normalized_resource_key = _build_resource_key(case_name, volume, reporter, page, year)
            return CitationCandidate(
                citation_text=citation_text,
                span=CitationSpan(start=start, end=start + len(citation_text)),
                case_name=case_name,
                volume=volume,
                reporter=reporter,
                page=page,
                pin_cite=pin_cite,
                court=court,
                year=year,
                citation_kind="full_case",
                parse_status="full_case_parsed",
                normalized_resource_key=normalized_resource_key,
            )

        return CitationCandidate(
            citation_text=citation_text,
            span=CitationSpan(start=start, end=end),
            case_name=None,
            volume=None,
            reporter=None,
            page=None,
            pin_cite=_normalize_pin_cite(getattr(getattr(citation, "metadata", None), "pin_cite", None)),
            court=None,
            year=None,
            citation_kind=citation_kind,
            parse_status="unsupported_reference_type",
            normalized_resource_key=None,
        )


def _citation_kind(citation: Any) -> str:
    type_name = type(citation).__name__
    if type_name == "FullCaseCitation":
        return "full_case"
    if type_name == "ShortCaseCitation":
        return "short_case"
    if type_name == "SupraCitation":
        return "supra"
    if type_name == "IdCitation":
        return "id"
    return "unsupported"


def _citation_span(citation: Any) -> tuple[int | None, int | None]:
    full_start = getattr(citation, "full_span_start", None)
    full_end = getattr(citation, "full_span_end", None)
    if full_start is not None and full_end is not None:
        return int(full_start), int(full_end)

    span = citation.span()
    if span is None:
        return None, None
    return int(span[0]), int(span[1])


def _build_case_name(plaintiff: str | None, defendant: str | None) -> str | None:
    if plaintiff and defendant:
        return normalize_text(f"{plaintiff} v. {defendant}")
    return None


def _normalize_pin_cite(value: str | None) -> str | None:
    if not value:
        return None
    normalized = normalize_text(value)
    return normalized.removeprefix("at ").strip() or None


def _fallback_full_case_candidates(
    text: str,
    existing_candidates: list[CitationCandidate],
) -> list[CitationCandidate]:
    candidates: list[CitationCandidate] = []
    existing_spans = {(candidate.span.start, candidate.span.end) for candidate in existing_candidates}
    for match in FULL_CASE_CITATION_RE.finditer(text):
        span = (match.start(), match.end())
        if span in existing_spans:
            continue
        court, year = _parse_parenthetical(match.group("parenthetical"))
        case_name = normalize_text(match.group("case_name"))
        volume = match.group("volume")
        reporter = normalize_text(match.group("reporter"))
        page = match.group("page")
        citation_text, start = _trim_leading_noncitation_lines(match.group(0), match.start())
        candidates.append(
            CitationCandidate(
                citation_text=citation_text,
                span=CitationSpan(start=start, end=start + len(citation_text)),
                case_name=case_name,
                volume=volume,
                reporter=reporter,
                page=page,
                pin_cite=_normalize_pin_cite(match.group("pin_cite")),
                court=court,
                year=year,
                citation_kind="full_case",
                parse_status="full_case_parsed",
                normalized_resource_key=_build_resource_key(case_name, volume, reporter, page, year),
            )
        )
    return candidates


def _trim_leading_noncitation_lines(raw_text: str, start: int) -> tuple[str, int]:
    offset = 0
    for line in raw_text.splitlines(keepends=True):
        if " v. " in line:
            trimmed = normalize_text(raw_text[offset:]).strip(" ,.;:")
            return trimmed, start + offset
        offset += len(line)
    return normalize_text(raw_text).strip(" ,.;:"), start


def _dedupe_candidates(candidates: list[CitationCandidate]) -> list[CitationCandidate]:
    seen: set[tuple[int, int, str]] = set()
    unique: list[CitationCandidate] = []
    for candidate in candidates:
        key = (candidate.span.start, candidate.span.end, candidate.citation_text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _parse_parenthetical(value: str | None) -> tuple[str | None, int | None]:
    if not value:
        return None, None

    normalized = normalize_text(value)
    match = PARENTHETICAL_YEAR_RE.fullmatch(normalized)
    if match is None:
        return normalized, None

    court = match.group("court")
    if court:
        court = court.strip(" ,;")
    year = int(match.group("year"))
    return court or None, year


def _normalize_court(value: str | None) -> str | None:
    if not value or value == "scotus":
        return None
    return normalize_text(value)


def _parse_year(value: str | int | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_resource_key(
    case_name: str | None,
    volume: str | None,
    reporter: str | None,
    page: str | None,
    year: int | None,
) -> str | None:
    if not case_name:
        return None

    parts = [_normalize_key_part(case_name)]
    if volume and reporter and page:
        parts.extend([volume.strip(), _normalize_reporter_key(reporter), page.strip()])
        if year is not None:
            parts.append(str(year))
    return "|".join(parts)


def _normalize_key_part(value: str) -> str:
    return " ".join(
        "".join(character if character.isalnum() else " " for character in normalize_text(value).lower()).split()
    ).replace(" v ", " v ")


def _normalize_reporter_key(value: str) -> str:
    return normalize_text(value).lower().replace(" ", "")
