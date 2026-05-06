"""Persisted authority identity lookup cache."""

from datetime import datetime

from sqlalchemy import DateTime, Integer, JSON, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, IdMixin


class AuthorityLookupCacheResult(Base, IdMixin):
    """Auditable identity-only cache entry for external authority lookup."""

    __tablename__ = "authority_lookup_cache_results"
    __table_args__ = (
        UniqueConstraint("provider", "lookup_key", name="uq_authority_lookup_cache_provider_lookup_key"),
    )

    provider: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    lookup_key: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    normalized_resource_key: Mapped[str | None] = mapped_column(String(512), index=True)
    volume: Mapped[str | None] = mapped_column(String(32))
    reporter: Mapped[str | None] = mapped_column(String(64))
    page: Mapped[str | None] = mapped_column(String(32))
    case_name: Mapped[str | None] = mapped_column(String(512))
    year: Mapped[int | None] = mapped_column(Integer)
    lookup_status: Mapped[str] = mapped_column(String(64), nullable=False)
    matched_provider_cluster_id: Mapped[str | None] = mapped_column(String(128))
    matched_case_name: Mapped[str | None] = mapped_column(String(512))
    matched_canonical_citation: Mapped[str | None] = mapped_column(String(512))
    matched_absolute_url: Mapped[str | None] = mapped_column(String(1024))
    matched_date_filed: Mapped[str | None] = mapped_column(String(32))
    matched_year: Mapped[int | None] = mapped_column(Integer)
    normalized_citations: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    raw_lookup_payload: Mapped[dict[str, object] | None] = mapped_column(JSON)
    error_message: Mapped[str | None] = mapped_column(Text)
    looked_up_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
