from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.models import AuthorityLookupCacheResult, Base
from app.services.authority.courtlistener import CourtListenerAuthorityLookupAdapter
from app.services.authority.lookup_cache import CachedAuthorityLookupService
from app.services.parsing.citation_extraction import CitationCandidate, CitationExtractionService, CitationSpan


@pytest.fixture
def session() -> Session:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    db_session = SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()
        engine.dispose()


def test_courtlistener_lookup_returns_authority_found_for_matching_case_identity() -> None:
    adapter = CourtListenerAuthorityLookupAdapter(
        settings=Settings(courtlistener_api_token="test-token"),
        transport=_transport_with(_found_payload(case_name="Brown v. Board of Education", year="1954")),
    )
    citation = _candidate("Brown v. Board of Education, 347 U.S. 483 (1954)")

    result = adapter.lookup(citation)

    assert result.lookup_status == "authority_found"
    assert result.provider == "courtlistener"
    assert result.source_name == "courtlistener_citation_lookup"
    assert result.matched_authority is not None
    assert result.matched_authority.provider_cluster_id == "1001"
    assert result.matched_authority.case_name == "Brown v. Board of Education"
    assert result.matched_authority.year == 1954
    assert result.normalized_citations == ["347 U.S. 483"]


def test_courtlistener_lookup_returns_authority_not_found_for_404_result() -> None:
    adapter = CourtListenerAuthorityLookupAdapter(
        settings=Settings(courtlistener_api_token="test-token"),
        transport=_transport_with(
            [
                {
                    "citation": "999 U.S. 1",
                    "normalized_citations": ["999 U.S. 1"],
                    "status": 404,
                    "error_message": "Citation not found.",
                    "clusters": [],
                }
            ]
        ),
    )
    citation = _candidate("Brown v. Davis, 999 U.S. 1 (2001)")

    result = adapter.lookup(citation)

    assert result.lookup_status == "authority_not_found"
    assert result.matched_authority is None
    assert result.error_message == "Citation not found."


def test_courtlistener_lookup_returns_name_mismatch_when_reporter_citation_belongs_to_different_case() -> None:
    adapter = CourtListenerAuthorityLookupAdapter(
        settings=Settings(courtlistener_api_token="test-token"),
        transport=_transport_with(_found_payload(case_name="Different v. Case", year="1954")),
    )
    citation = _candidate("Brown v. Board of Education, 347 U.S. 483 (1954)")

    result = adapter.lookup(citation)

    assert result.lookup_status == "authority_name_mismatch"
    assert result.matched_authority is not None
    assert result.matched_authority.case_name == "Different v. Case"


def test_courtlistener_lookup_returns_year_mismatch_when_reporter_citation_year_conflicts() -> None:
    adapter = CourtListenerAuthorityLookupAdapter(
        settings=Settings(courtlistener_api_token="test-token"),
        transport=_transport_with(_found_payload(case_name="Brown v. Board of Education", year="1955")),
    )
    citation = _candidate("Brown v. Board of Education, 347 U.S. 483 (1954)")

    result = adapter.lookup(citation)

    assert result.lookup_status == "authority_year_mismatch"
    assert result.matched_authority is not None
    assert result.matched_authority.year == 1955


def test_courtlistener_lookup_returns_missing_fields_before_network_call() -> None:
    def fail_transport(**_: Any) -> list[dict[str, Any]]:
        raise AssertionError("transport should not be called when reporter fields are missing")

    adapter = CourtListenerAuthorityLookupAdapter(
        settings=Settings(courtlistener_api_token="test-token"),
        transport=fail_transport,
    )
    citation = CitationCandidate(
        citation_text="Smith v. Jones",
        span=CitationSpan(start=0, end=14),
        case_name="Smith v. Jones",
        volume=None,
        reporter=None,
        page=None,
        pin_cite=None,
        court=None,
        year=None,
        citation_kind="full_case",
        parse_status="full_case_parsed",
        normalized_resource_key="smith v jones",
    )

    result = adapter.lookup(citation)

    assert result.lookup_status == "lookup_missing_fields"
    assert result.matched_authority is None
    assert "volume" in (result.error_message or "")


def test_courtlistener_lookup_returns_not_attempted_without_token() -> None:
    adapter = CourtListenerAuthorityLookupAdapter(settings=Settings(courtlistener_api_token=None))
    citation = _candidate("Brown v. Board of Education, 347 U.S. 483 (1954)")

    result = adapter.lookup(citation)

    assert result.lookup_status == "lookup_not_attempted"
    assert result.provider is None


def test_cached_authority_lookup_persists_first_live_lookup_result(session: Session) -> None:
    service = CachedAuthorityLookupService(
        session,
        live_lookup=CourtListenerAuthorityLookupAdapter(
            settings=Settings(courtlistener_api_token="test-token"),
            transport=_transport_with(_found_payload(case_name="Brown v. Board of Education", year="1954")),
        ),
    )
    citation = _candidate("Brown v. Board of Education, 347 U.S. 483 (1954)")

    result = service.lookup(citation)

    assert result.lookup_status == "authority_found"
    assert result.cached is False
    cache_records = session.query(AuthorityLookupCacheResult).all()
    assert len(cache_records) == 1
    assert cache_records[0].provider == "courtlistener"
    assert cache_records[0].lookup_key == "brown v board of education|347|u.s.|483|1954"
    assert cache_records[0].lookup_status == "authority_found"
    assert cache_records[0].matched_case_name == "Brown v. Board of Education"
    assert cache_records[0].raw_lookup_payload is not None


def test_cached_authority_lookup_reuses_cached_result_without_second_live_call(session: Session) -> None:
    live_call_count = 0

    def counting_transport(**_: Any) -> list[dict[str, Any]]:
        nonlocal live_call_count
        live_call_count += 1
        return _found_payload(case_name="Brown v. Board of Education", year="1954")

    service = CachedAuthorityLookupService(
        session,
        live_lookup=CourtListenerAuthorityLookupAdapter(
            settings=Settings(courtlistener_api_token="test-token"),
            transport=counting_transport,
        ),
    )
    citation = _candidate("Brown v. Board of Education, 347 U.S. 483 (1954)")

    first = service.lookup(citation)
    second = service.lookup(citation)

    assert first.cached is False
    assert second.cached is True
    assert second.lookup_status == "authority_found"
    assert second.matched_authority is not None
    assert second.matched_authority.case_name == "Brown v. Board of Education"
    assert live_call_count == 1
    assert session.query(AuthorityLookupCacheResult).count() == 1


def test_cached_authority_lookup_persists_not_found_result(session: Session) -> None:
    service = CachedAuthorityLookupService(
        session,
        live_lookup=CourtListenerAuthorityLookupAdapter(
            settings=Settings(courtlistener_api_token="test-token"),
            transport=_transport_with(
                [
                    {
                        "citation": "999 U.S. 1",
                        "normalized_citations": ["999 U.S. 1"],
                        "status": 404,
                        "error_message": "Citation not found.",
                        "clusters": [],
                    }
                ]
            ),
        ),
    )
    citation = _candidate("Brown v. Davis, 999 U.S. 1 (2001)")

    result = service.lookup(citation)

    assert result.lookup_status == "authority_not_found"
    record = session.query(AuthorityLookupCacheResult).one()
    assert record.lookup_status == "authority_not_found"
    assert record.error_message == "Citation not found."


def test_cached_authority_lookup_persists_name_mismatch_result(session: Session) -> None:
    service = CachedAuthorityLookupService(
        session,
        live_lookup=CourtListenerAuthorityLookupAdapter(
            settings=Settings(courtlistener_api_token="test-token"),
            transport=_transport_with(_found_payload(case_name="Different v. Case", year="1954")),
        ),
    )
    citation = _candidate("Brown v. Board of Education, 347 U.S. 483 (1954)")

    result = service.lookup(citation)

    assert result.lookup_status == "authority_name_mismatch"
    record = session.query(AuthorityLookupCacheResult).one()
    assert record.lookup_status == "authority_name_mismatch"
    assert record.matched_case_name == "Different v. Case"


def test_cached_authority_lookup_does_not_cache_not_attempted_without_token(session: Session) -> None:
    service = CachedAuthorityLookupService(
        session,
        live_lookup=CourtListenerAuthorityLookupAdapter(settings=Settings(courtlistener_api_token=None)),
    )
    citation = _candidate("Brown v. Board of Education, 347 U.S. 483 (1954)")

    result = service.lookup(citation)

    assert result.lookup_status == "lookup_not_attempted"
    assert session.query(AuthorityLookupCacheResult).count() == 0


def _candidate(text: str) -> CitationCandidate:
    candidate = CitationExtractionService().parse_full_case_citation(text)
    assert candidate is not None
    return candidate


def _transport_with(payload: list[dict[str, Any]]):
    def transport(**_: Any) -> list[dict[str, Any]]:
        return payload

    return transport


def _found_payload(*, case_name: str, year: str) -> list[dict[str, Any]]:
    return [
        {
            "citation": "347 U.S. 483",
            "normalized_citations": ["347 U.S. 483"],
            "status": 200,
            "error_message": "",
            "clusters": [
                {
                    "id": 1001,
                    "case_name": case_name,
                    "date_filed": f"{year}-05-17",
                    "absolute_url": "/opinion/1001/brown-v-board-of-education/",
                    "citations": [{"volume": 347, "reporter": "U.S.", "page": "483"}],
                }
            ],
        }
    ]
