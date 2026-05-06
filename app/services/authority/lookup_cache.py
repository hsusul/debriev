"""DB-backed authority identity lookup cache orchestration."""

from sqlalchemy.orm import Session

from app.models import AuthorityLookupCacheResult
from app.repositories.authority_lookup_cache import AuthorityLookupCacheRepository
from app.services.authority.courtlistener import (
    AuthorityIdentityLookupResult,
    CourtListenerAuthorityLookupAdapter,
    ExternalAuthorityIdentity,
)
from app.services.parsing.citation_extraction import CitationCandidate
from app.services.parsing.normalization import normalize_text


class CachedAuthorityLookupService:
    """Reuse persisted identity lookups before calling the external adapter."""

    def __init__(
        self,
        session: Session,
        *,
        live_lookup: CourtListenerAuthorityLookupAdapter | None = None,
    ) -> None:
        self.repository = AuthorityLookupCacheRepository(session)
        self.live_lookup = live_lookup or CourtListenerAuthorityLookupAdapter()

    def lookup(self, citation: CitationCandidate) -> AuthorityIdentityLookupResult:
        lookup_key = build_authority_lookup_key(citation)
        cached = self.repository.get_by_lookup_key(provider=self.live_lookup.provider, lookup_key=lookup_key)
        if cached is not None:
            return _result_from_cache(cached)

        result = self.live_lookup.lookup(citation)
        if result.lookup_status != "lookup_not_attempted":
            self.repository.create(
                provider=self.live_lookup.provider,
                lookup_key=lookup_key,
                normalized_resource_key=citation.normalized_resource_key,
                volume=citation.volume,
                reporter=citation.reporter,
                page=citation.page,
                case_name=citation.case_name,
                year=citation.year,
                lookup_status=result.lookup_status,
                matched_provider_cluster_id=(
                    result.matched_authority.provider_cluster_id if result.matched_authority is not None else None
                ),
                matched_case_name=result.matched_authority.case_name if result.matched_authority is not None else None,
                matched_canonical_citation=(
                    result.matched_authority.canonical_citation if result.matched_authority is not None else None
                ),
                matched_absolute_url=result.matched_authority.absolute_url if result.matched_authority is not None else None,
                matched_date_filed=result.matched_authority.date_filed if result.matched_authority is not None else None,
                matched_year=result.matched_authority.year if result.matched_authority is not None else None,
                normalized_citations=result.normalized_citations,
                raw_lookup_payload=result.raw_lookup_payload,
                error_message=result.error_message,
            )
        return result


def build_authority_lookup_key(citation: CitationCandidate) -> str:
    if citation.normalized_resource_key:
        return citation.normalized_resource_key

    parts = [
        citation.case_name or "",
        citation.volume or "",
        citation.reporter or "",
        citation.page or "",
        str(citation.year) if citation.year is not None else "",
    ]
    return "|".join(normalize_text(part).lower().strip() for part in parts)


def _result_from_cache(record: AuthorityLookupCacheResult) -> AuthorityIdentityLookupResult:
    matched_authority = None
    if record.matched_provider_cluster_id or record.matched_case_name or record.matched_canonical_citation:
        matched_authority = ExternalAuthorityIdentity(
            provider=record.provider,
            provider_cluster_id=record.matched_provider_cluster_id,
            case_name=record.matched_case_name,
            canonical_citation=record.matched_canonical_citation,
            absolute_url=record.matched_absolute_url,
            date_filed=record.matched_date_filed,
            year=record.matched_year,
            normalized_citations=list(record.normalized_citations or []),
        )
    return AuthorityIdentityLookupResult(
        lookup_status=record.lookup_status,
        provider=record.provider,
        source_name=f"{record.provider}_citation_lookup",
        matched_authority=matched_authority,
        normalized_citations=list(record.normalized_citations or []),
        raw_lookup_payload=record.raw_lookup_payload,
        error_message=record.error_message,
        cached=True,
    )
