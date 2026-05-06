"""Identity-only CourtListener lookup for parsed full case citations."""

from dataclasses import dataclass
import json
import re
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.config import Settings, get_settings
from app.services.parsing.citation_extraction import CitationCandidate
from app.services.parsing.normalization import normalize_for_match, normalize_text

COURTLISTENER_CITATION_LOOKUP_PATH = "/api/rest/v4/citation-lookup/"
NON_ALPHANUMERIC_RE = re.compile(r"[^a-z0-9]+")


class CourtListenerTransport(Protocol):
    def __call__(
        self,
        *,
        url: str,
        token: str,
        data: dict[str, str],
        timeout_seconds: float,
    ) -> list[dict[str, Any]]:
        """Submit a CourtListener citation lookup request."""


@dataclass(slots=True, frozen=True)
class ExternalAuthorityIdentity:
    provider: str
    provider_cluster_id: str | None
    case_name: str | None
    canonical_citation: str | None
    absolute_url: str | None
    date_filed: str | None
    year: int | None
    normalized_citations: list[str]


@dataclass(slots=True, frozen=True)
class AuthorityIdentityLookupResult:
    lookup_status: str
    provider: str | None
    source_name: str | None
    matched_authority: ExternalAuthorityIdentity | None
    normalized_citations: list[str]
    raw_lookup_payload: dict[str, object] | None
    error_message: str | None
    cached: bool = False


class CourtListenerAuthorityLookupAdapter:
    """Look up case identity by citation without retrieving opinion content."""

    provider = "courtlistener"
    source_name = "courtlistener_citation_lookup"

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        transport: CourtListenerTransport | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.transport = transport or _post_courtlistener_lookup

    def lookup(self, citation: CitationCandidate) -> AuthorityIdentityLookupResult:
        if not self.settings.courtlistener_api_token:
            return _not_attempted("CourtListener lookup not attempted because COURTLISTENER_API_TOKEN is not configured.")

        missing_fields = [
            field_name
            for field_name, value in (
                ("volume", citation.volume),
                ("reporter", citation.reporter),
                ("page", citation.page),
            )
            if not value
        ]
        if missing_fields:
            return AuthorityIdentityLookupResult(
                lookup_status="lookup_missing_fields",
                provider=self.provider,
                source_name=self.source_name,
                matched_authority=None,
                normalized_citations=[],
                raw_lookup_payload=None,
                error_message=f"Missing required citation lookup fields: {', '.join(missing_fields)}.",
            )

        try:
            payload = self.transport(
                url=self._citation_lookup_url(),
                token=self.settings.courtlistener_api_token,
                data={
                    "volume": citation.volume or "",
                    "reporter": citation.reporter or "",
                    "page": citation.page or "",
                },
                timeout_seconds=self.settings.courtlistener_timeout_seconds,
            )
        except TimeoutError:
            return _unavailable("CourtListener citation lookup timed out.")
        except (HTTPError, URLError, OSError) as exc:
            return _unavailable(f"CourtListener citation lookup unavailable: {exc}")
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            return _error(f"CourtListener citation lookup returned an invalid response: {exc}")

        if not payload:
            return AuthorityIdentityLookupResult(
                lookup_status="authority_not_found",
                provider=self.provider,
                source_name=self.source_name,
                matched_authority=None,
                normalized_citations=[],
                raw_lookup_payload=None,
                error_message="CourtListener returned no citation lookup results.",
            )

        result = payload[0]
        normalized_citations = _normalized_citations(result)
        status = int(result.get("status") or 0)
        clusters = [cluster for cluster in result.get("clusters") or [] if isinstance(cluster, dict)]

        if status == 404 or not clusters:
            return AuthorityIdentityLookupResult(
                lookup_status="authority_not_found",
                provider=self.provider,
                source_name=self.source_name,
                matched_authority=None,
                normalized_citations=normalized_citations,
                raw_lookup_payload=_raw_payload(result),
                error_message=_error_message(result) or "Citation was not found in CourtListener.",
            )
        if status == 429:
            return _unavailable(_error_message(result) or "CourtListener citation lookup was throttled.")
        if status >= 500:
            return _unavailable(_error_message(result) or "CourtListener citation lookup failed upstream.")
        if status >= 400:
            return _error(_error_message(result) or "CourtListener citation lookup rejected the citation.")

        year_compatible = [cluster for cluster in clusters if _year_matches(citation.year, _cluster_year(cluster))]
        if citation.year is not None and clusters and not year_compatible:
            return AuthorityIdentityLookupResult(
                lookup_status="authority_year_mismatch",
                provider=self.provider,
                source_name=self.source_name,
                matched_authority=_external_authority_from_cluster(clusters[0], normalized_citations),
                normalized_citations=normalized_citations,
                raw_lookup_payload=_raw_payload(result),
                error_message="CourtListener found the reporter citation, but the returned case year did not match.",
            )

        candidate_clusters = year_compatible or clusters
        name_compatible = [
            cluster
            for cluster in candidate_clusters
            if _case_name_matches(citation.case_name, _cluster_case_name(cluster))
        ]
        if citation.case_name and candidate_clusters and not name_compatible:
            return AuthorityIdentityLookupResult(
                lookup_status="authority_name_mismatch",
                provider=self.provider,
                source_name=self.source_name,
                matched_authority=_external_authority_from_cluster(candidate_clusters[0], normalized_citations),
                normalized_citations=normalized_citations,
                raw_lookup_payload=_raw_payload(result),
                error_message="CourtListener found the reporter citation, but the returned case name did not match.",
            )

        matched_cluster = (name_compatible or candidate_clusters)[0]
        return AuthorityIdentityLookupResult(
            lookup_status="authority_found",
            provider=self.provider,
            source_name=self.source_name,
            matched_authority=_external_authority_from_cluster(matched_cluster, normalized_citations),
            normalized_citations=normalized_citations,
            raw_lookup_payload=_raw_payload(result),
            error_message=None,
        )

    def _citation_lookup_url(self) -> str:
        base_url = self.settings.courtlistener_base_url.rstrip("/")
        return f"{base_url}{COURTLISTENER_CITATION_LOOKUP_PATH}"


def _post_courtlistener_lookup(
    *,
    url: str,
    token: str,
    data: dict[str, str],
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    encoded = urlencode(data).encode("utf-8")
    request = Request(
        url,
        data=encoded,
        headers={
            "Authorization": f"Token {token}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Expected CourtListener citation lookup response to be a list.")
    return payload


def _not_attempted(message: str) -> AuthorityIdentityLookupResult:
    return AuthorityIdentityLookupResult(
        lookup_status="lookup_not_attempted",
        provider=None,
        source_name=None,
        matched_authority=None,
        normalized_citations=[],
        raw_lookup_payload=None,
        error_message=message,
    )


def _unavailable(message: str) -> AuthorityIdentityLookupResult:
    return AuthorityIdentityLookupResult(
        lookup_status="lookup_unavailable",
        provider="courtlistener",
        source_name="courtlistener_citation_lookup",
        matched_authority=None,
        normalized_citations=[],
        raw_lookup_payload=None,
        error_message=message,
    )


def _error(message: str) -> AuthorityIdentityLookupResult:
    return AuthorityIdentityLookupResult(
        lookup_status="lookup_error",
        provider="courtlistener",
        source_name="courtlistener_citation_lookup",
        matched_authority=None,
        normalized_citations=[],
        raw_lookup_payload=None,
        error_message=message,
    )


def _raw_payload(result: dict[str, Any]) -> dict[str, object]:
    return dict(result)


def _external_authority_from_cluster(
    cluster: dict[str, Any],
    normalized_citations: list[str],
) -> ExternalAuthorityIdentity:
    date_filed = _as_string(cluster.get("date_filed"))
    return ExternalAuthorityIdentity(
        provider="courtlistener",
        provider_cluster_id=_as_string(cluster.get("id") or cluster.get("cluster_id")),
        case_name=_cluster_case_name(cluster),
        canonical_citation=_canonical_citation(cluster, normalized_citations),
        absolute_url=_as_string(cluster.get("absolute_url")),
        date_filed=date_filed,
        year=_cluster_year(cluster),
        normalized_citations=normalized_citations,
    )


def _normalized_citations(result: dict[str, Any]) -> list[str]:
    values = result.get("normalized_citations")
    if not isinstance(values, list):
        return []
    return [normalize_text(str(value)) for value in values if value]


def _error_message(result: dict[str, Any]) -> str | None:
    value = result.get("error_message")
    if not value:
        return None
    return normalize_text(str(value))


def _cluster_case_name(cluster: dict[str, Any]) -> str | None:
    for key in ("case_name", "case_name_full", "case_name_short"):
        value = cluster.get(key)
        if value:
            return normalize_text(str(value))
    return None


def _cluster_year(cluster: dict[str, Any]) -> int | None:
    explicit_year = _parse_year(cluster.get("year"))
    if explicit_year is not None:
        return explicit_year
    date_filed = _as_string(cluster.get("date_filed"))
    if date_filed and len(date_filed) >= 4:
        return _parse_year(date_filed[:4])
    return None


def _canonical_citation(cluster: dict[str, Any], normalized_citations: list[str]) -> str | None:
    citations = cluster.get("citations")
    if isinstance(citations, list) and citations:
        first = citations[0]
        if isinstance(first, dict):
            volume = _as_string(first.get("volume"))
            reporter = _as_string(first.get("reporter"))
            page = _as_string(first.get("page"))
            if volume and reporter and page:
                return f"{volume} {reporter} {page}"
    if normalized_citations:
        return normalized_citations[0]
    return None


def _case_name_matches(expected: str | None, actual: str | None) -> bool:
    if not expected:
        return True
    if not actual:
        return False

    expected_key = _normalize_name(expected)
    actual_key = _normalize_name(actual)
    if expected_key == actual_key:
        return True

    expected_parts = _party_name_parts(expected_key)
    actual_parts = _party_name_parts(actual_key)
    if not expected_parts or not actual_parts:
        return False
    return expected_parts == actual_parts


def _year_matches(expected: int | None, actual: int | None) -> bool:
    return expected is None or actual is None or expected == actual


def _party_name_parts(value: str) -> tuple[str, str] | None:
    pieces = value.split(" v ")
    if len(pieces) != 2:
        return None
    return pieces[0].strip(), pieces[1].strip()


def _normalize_name(value: str) -> str:
    normalized = normalize_for_match(value).replace(" v. ", " v ")
    return NON_ALPHANUMERIC_RE.sub(" ", normalized).strip()


def _parse_year(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_string(value: Any) -> str | None:
    if value is None:
        return None
    return normalize_text(str(value))
