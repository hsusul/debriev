from app.core.enums import SupportStatus
from app.services.parsing.case_citations import parse_case_citation, resolve_case_authority
from app.services.verification.authority_content import (
    evaluate_proposition_against_authority_content,
    get_authority_content,
)


def test_get_authority_content_returns_repo_local_excerpt_catalog_for_supported_case() -> None:
    matched = resolve_case_authority(parse_case_citation("Brown v. Board of Education, 347 U.S. 483 (1954)"))

    assert matched is not None
    content = get_authority_content(matched)
    assert content is not None
    assert content.authority_id == "brown-v-board-of-education-347-us-483"
    assert len(content.excerpts) >= 1


def test_get_authority_content_returns_none_when_catalog_has_no_case_text() -> None:
    matched = resolve_case_authority(parse_case_citation("Anderson v. Liberty Lobby, Inc., 477 U.S. 242 (1986)"))

    assert matched is not None
    assert get_authority_content(matched) is None


def test_evaluate_proposition_against_authority_content_marks_supported_overlap() -> None:
    matched = resolve_case_authority(parse_case_citation("Brown v. Board of Education, 347 U.S. 483 (1954)"))

    assert matched is not None
    evaluation = evaluate_proposition_against_authority_content(
        "Brown v. Board of Education, 347 U.S. 483 (1954), held that segregation in public schools deprives children of equal educational opportunities.",
        "Brown v. Board of Education, 347 U.S. 483 (1954)",
        matched,
    )

    assert evaluation.authority_content_status == "support_verified"
    assert evaluation.proposition_verdict == SupportStatus.SUPPORTED
    assert evaluation.authority_excerpt is not None
    assert evaluation.support_verification_basis == "authority_content_deterministic_supported"


def test_evaluate_proposition_against_authority_content_marks_unavailable_when_no_text_exists() -> None:
    matched = resolve_case_authority(parse_case_citation("Anderson v. Liberty Lobby, Inc., 477 U.S. 242 (1986)"))

    assert matched is not None
    evaluation = evaluate_proposition_against_authority_content(
        "Anderson v. Liberty Lobby, Inc., 477 U.S. 242 (1986), held that summary judgment always fails when facts are disputed.",
        "Anderson v. Liberty Lobby, Inc., 477 U.S. 242 (1986)",
        matched,
    )

    assert evaluation.authority_content_status == "unavailable"
    assert evaluation.proposition_verdict is None
    assert evaluation.support_verification_basis == "authority_content_unavailable"


def test_evaluate_proposition_against_authority_content_marks_unsupported_when_overlap_is_missing() -> None:
    matched = resolve_case_authority(parse_case_citation("Brown v. Board of Education, 347 U.S. 483 (1954)"))

    assert matched is not None
    evaluation = evaluate_proposition_against_authority_content(
        "Brown v. Board of Education, 347 U.S. 483 (1954), held that negligence is always enough.",
        "Brown v. Board of Education, 347 U.S. 483 (1954)",
        matched,
    )

    assert evaluation.authority_content_status == "available"
    assert evaluation.proposition_verdict == SupportStatus.UNSUPPORTED
    assert evaluation.support_verification_basis == "authority_content_deterministic_no_clear_support"
