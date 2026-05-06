"""Repository helpers for authority lookup cache results."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AuthorityLookupCacheResult


class AuthorityLookupCacheRepository:
    """Persistence helpers for identity-only authority lookup cache entries."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_lookup_key(self, *, provider: str, lookup_key: str) -> AuthorityLookupCacheResult | None:
        stmt = select(AuthorityLookupCacheResult).where(
            AuthorityLookupCacheResult.provider == provider,
            AuthorityLookupCacheResult.lookup_key == lookup_key,
        )
        return self.session.scalars(stmt).one_or_none()

    def create(
        self,
        *,
        provider: str,
        lookup_key: str,
        normalized_resource_key: str | None,
        volume: str | None,
        reporter: str | None,
        page: str | None,
        case_name: str | None,
        year: int | None,
        lookup_status: str,
        matched_provider_cluster_id: str | None,
        matched_case_name: str | None,
        matched_canonical_citation: str | None,
        matched_absolute_url: str | None,
        matched_date_filed: str | None,
        matched_year: int | None,
        normalized_citations: list[str],
        raw_lookup_payload: dict[str, object] | None,
        error_message: str | None,
    ) -> AuthorityLookupCacheResult:
        record = AuthorityLookupCacheResult(
            provider=provider,
            lookup_key=lookup_key,
            normalized_resource_key=normalized_resource_key,
            volume=volume,
            reporter=reporter,
            page=page,
            case_name=case_name,
            year=year,
            lookup_status=lookup_status,
            matched_provider_cluster_id=matched_provider_cluster_id,
            matched_case_name=matched_case_name,
            matched_canonical_citation=matched_canonical_citation,
            matched_absolute_url=matched_absolute_url,
            matched_date_filed=matched_date_filed,
            matched_year=matched_year,
            normalized_citations=normalized_citations,
            raw_lookup_payload=raw_lookup_payload,
            error_message=error_message,
        )
        self.session.add(record)
        self.session.flush()
        return record
