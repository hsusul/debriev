from app.services.parsing.case_citations import parse_case_citation, resolve_case_authority
from app.services.workflows.citation_verification import _derive_authority_match_status, _derive_authority_status


def test_parse_case_citation_extracts_structured_authority_fields() -> None:
    parsed = parse_case_citation("Brown v. Board of Education, 347 U.S. 483 (1954)")

    assert parsed.case_name == "Brown v. Board of Education"
    assert parsed.reporter_volume == "347"
    assert parsed.reporter_abbreviation == "U.S."
    assert parsed.first_page == "483"
    assert parsed.court is None
    assert parsed.year == 1954
    assert parsed.normalized_authority_reference == "brown v board of education|347|u.s.|483|1954"


def test_resolve_case_authority_matches_known_mvp_catalog_entry() -> None:
    parsed = parse_case_citation("Brown v. Board of Education, 347 U.S. 483 (1954)")
    matched = resolve_case_authority(parsed)

    assert matched is not None
    assert matched.authority_id == "brown-v-board-of-education-347-us-483"
    assert matched.canonical_citation == "Brown v. Board of Education, 347 U.S. 483 (1954)"


def test_resolve_case_authority_returns_none_for_unmatched_candidate() -> None:
    parsed = parse_case_citation("Brown v. Davis, 999 U.S. 1 (2001)")

    assert resolve_case_authority(parsed) is None
    assert _derive_authority_status(parsed, None, has_linked_support=False) == "authority_candidate_parsed"
    assert _derive_authority_match_status(parsed, None) == "no_match"


def test_authority_status_prefers_linked_support_when_support_snapshot_exists() -> None:
    parsed = parse_case_citation("Brown v. Board of Education, 347 U.S. 483 (1954)")
    matched = resolve_case_authority(parsed)

    assert matched is not None
    assert _derive_authority_status(parsed, matched, has_linked_support=True) == "linked_authority_support_present"
    assert _derive_authority_match_status(parsed, matched) == "matched"
