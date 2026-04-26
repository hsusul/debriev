"""Deterministic case citation parsing and limited MVP authority matching."""

from dataclasses import dataclass
import re

from app.services.parsing.normalization import normalize_text

SIGNAL_PREFIX_RE = re.compile(r"^(Under|See|But see|Cf\.)\s+", re.IGNORECASE)
CASE_NAME_TOKEN = r"[A-Z][A-Za-z0-9'’.&-]*"
CASE_NAME_CONNECTOR = r"(?:of|and|the|for|in|on|at|to|by|ex rel\.)"
CASE_NAME_WORD = rf"(?:{CASE_NAME_TOKEN}|{CASE_NAME_CONNECTOR})"
CASE_NAME_SUFFIX = r"(?:,\s+(?:Inc\.|Corp\.|Co\.|Ltd\.|LLC))?"
CASE_NAME_PARTY = rf"{CASE_NAME_TOKEN}(?:\s+{CASE_NAME_WORD}){{0,7}}{CASE_NAME_SUFFIX}"
REPORTER_TOKEN = r"[A-Z][A-Za-z0-9.\-]*"
REPORTER_PART = rf"{REPORTER_TOKEN}(?:\s+{REPORTER_TOKEN})*"
CASE_CITATION_RE = re.compile(
    rf"\b(?:(?P<signal>Under|See|But see|Cf\.)\s+)?"
    rf"(?P<case_name>{CASE_NAME_PARTY}\s+v\.\s+{CASE_NAME_PARTY})"
    rf"(?:,\s*(?P<volume>\d{{1,4}})\s+(?P<reporter>{REPORTER_PART})\s+(?P<page>\d{{1,5}}))?"
    rf"(?:\s*\((?P<parenthetical>[^()]*?\d{{4}})\))?"
)
PARENTHETICAL_YEAR_RE = re.compile(r"^(?:(?P<court>.*\S)\s+)?(?P<year>\d{4})$")
NON_ALPHANUMERIC_RE = re.compile(r"[^a-z0-9]+")


@dataclass(slots=True, frozen=True)
class ParsedCaseCitation:
    citation_text: str
    normalized_citation_text: str
    signal: str | None
    case_name: str | None
    reporter_volume: str | None
    reporter_abbreviation: str | None
    first_page: str | None
    court: str | None
    year: int | None

    @property
    def is_case_citation(self) -> bool:
        return self.case_name is not None

    @property
    def has_structured_authority_candidate(self) -> bool:
        return all(
            (
                self.case_name,
                self.reporter_volume,
                self.reporter_abbreviation,
                self.first_page,
            )
        )

    @property
    def normalized_authority_reference(self) -> str | None:
        if not self.case_name:
            return None

        parts = [_normalize_case_name_key(self.case_name)]
        if self.has_structured_authority_candidate:
            parts.extend(
                [
                    self.reporter_volume or "",
                    _normalize_reporter_key(self.reporter_abbreviation or ""),
                    self.first_page or "",
                ]
            )
            if self.year is not None:
                parts.append(str(self.year))
        return "|".join(parts)


@dataclass(slots=True, frozen=True)
class AuthorityCatalogEntry:
    authority_id: str
    case_name: str
    reporter_volume: str
    reporter_abbreviation: str
    first_page: str
    court: str | None
    year: int | None
    source_name: str

    @property
    def canonical_citation(self) -> str:
        citation = f"{self.case_name}, {self.reporter_volume} {self.reporter_abbreviation} {self.first_page}"
        if self.year is None and self.court is None:
            return citation
        if self.year is None:
            return f"{citation} ({self.court})"
        if self.court is None:
            return f"{citation} ({self.year})"
        return f"{citation} ({self.court} {self.year})"


@dataclass(slots=True, frozen=True)
class AuthorityMatch:
    authority_id: str
    canonical_name: str
    canonical_citation: str
    reporter_volume: str
    reporter_abbreviation: str
    first_page: str
    court: str | None
    year: int | None
    source_name: str


_AUTHORITY_CATALOG = (
    AuthorityCatalogEntry(
        authority_id="brown-v-board-of-education-347-us-483",
        case_name="Brown v. Board of Education",
        reporter_volume="347",
        reporter_abbreviation="U.S.",
        first_page="483",
        court=None,
        year=1954,
        source_name="debriev_mvp_authority_catalog",
    ),
    AuthorityCatalogEntry(
        authority_id="celotex-v-catrett-477-us-317",
        case_name="Celotex Corp. v. Catrett",
        reporter_volume="477",
        reporter_abbreviation="U.S.",
        first_page="317",
        court=None,
        year=1986,
        source_name="debriev_mvp_authority_catalog",
    ),
    AuthorityCatalogEntry(
        authority_id="anderson-v-liberty-lobby-477-us-242",
        case_name="Anderson v. Liberty Lobby, Inc.",
        reporter_volume="477",
        reporter_abbreviation="U.S.",
        first_page="242",
        court=None,
        year=1986,
        source_name="debriev_mvp_authority_catalog",
    ),
)

def extract_case_citation_texts(text: str) -> list[str]:
    """Return distinct case citation strings in encounter order."""

    seen: set[str] = set()
    citations: list[str] = []
    for match in CASE_CITATION_RE.finditer(text):
        citation_text = normalize_case_citation_text(match.group(0))
        if citation_text in seen:
            continue
        seen.add(citation_text)
        citations.append(citation_text)
    return citations


def parse_case_citation(text: str) -> ParsedCaseCitation:
    """Parse a raw citation string into structured authority fields."""

    citation_text = normalize_case_citation_text(text)
    match = CASE_CITATION_RE.fullmatch(citation_text)
    if match is None:
        return ParsedCaseCitation(
            citation_text=citation_text,
            normalized_citation_text=citation_text,
            signal=None,
            case_name=None,
            reporter_volume=None,
            reporter_abbreviation=None,
            first_page=None,
            court=None,
            year=None,
        )

    court, year = _parse_parenthetical(match.group("parenthetical"))
    reporter = _normalize_reporter_abbreviation(match.group("reporter"))
    return ParsedCaseCitation(
        citation_text=citation_text,
        normalized_citation_text=citation_text,
        signal=match.group("signal"),
        case_name=normalize_text(match.group("case_name")),
        reporter_volume=match.group("volume"),
        reporter_abbreviation=reporter,
        first_page=match.group("page"),
        court=court,
        year=year,
    )


def resolve_case_authority(parsed: ParsedCaseCitation) -> AuthorityMatch | None:
    """Resolve a parsed citation against the limited built-in MVP authority catalog."""

    if not parsed.has_structured_authority_candidate:
        return None

    reporter_key = _build_reporter_key(
        parsed.reporter_volume or "",
        parsed.reporter_abbreviation or "",
        parsed.first_page or "",
    )
    entry = _CATALOG_BY_REPORTER_KEY.get(reporter_key)
    if entry is None:
        return None
    if parsed.case_name is None:
        return None
    if _normalize_case_name_key(parsed.case_name) != _normalize_case_name_key(entry.case_name):
        return None
    if parsed.year is not None and entry.year is not None and parsed.year != entry.year:
        return None

    return AuthorityMatch(
        authority_id=entry.authority_id,
        canonical_name=entry.case_name,
        canonical_citation=entry.canonical_citation,
        reporter_volume=entry.reporter_volume,
        reporter_abbreviation=entry.reporter_abbreviation,
        first_page=entry.first_page,
        court=entry.court,
        year=entry.year,
        source_name=entry.source_name,
    )


def normalize_case_citation_text(text: str) -> str:
    """Remove leading signal text and trailing punctuation from a citation span."""

    normalized = normalize_text(text)
    normalized = SIGNAL_PREFIX_RE.sub("", normalized)
    return normalized.strip(" ,.;:")


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


def _build_reporter_key(volume: str, reporter_abbreviation: str, first_page: str) -> str:
    return "|".join([volume.strip(), _normalize_reporter_key(reporter_abbreviation), first_page.strip()])


def _normalize_reporter_key(value: str) -> str:
    return normalize_text(value).lower().replace(" ", "")


def _normalize_reporter_abbreviation(value: str | None) -> str | None:
    if not value:
        return None
    return normalize_text(value)


def _normalize_case_name_key(value: str) -> str:
    normalized = normalize_text(value).lower().replace(" v. ", " v ")
    return NON_ALPHANUMERIC_RE.sub(" ", normalized).strip()


_CATALOG_BY_REPORTER_KEY = {
    _build_reporter_key(
        entry.reporter_volume,
        entry.reporter_abbreviation,
        entry.first_page,
    ): entry
    for entry in _AUTHORITY_CATALOG
}
