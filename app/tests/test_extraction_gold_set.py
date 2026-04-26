import pytest

from app.services.claims.extractor import extract_claim_candidates
from app.tests.gold_eval_report import render_extraction_mismatch, render_suite_report, run_extraction_gold_evaluation
from app.tests.gold_sets import EXTRACTION_GOLD_CASES, ExtractionGoldCase


@pytest.mark.parametrize("case", EXTRACTION_GOLD_CASES, ids=lambda case: case.name)
def test_extraction_gold_set(case: ExtractionGoldCase) -> None:
    claims = extract_claim_candidates(case.assertion_text)
    actual_texts = tuple(claim.text for claim in claims)
    actual_types = tuple(claim.claim_type for claim in claims)

    assert actual_texts == case.expected_claim_texts, render_extraction_mismatch(
        case,
        actual_texts=actual_texts,
        actual_types=actual_types,
    )
    assert actual_types == case.expected_claim_types, render_extraction_mismatch(
        case,
        actual_texts=actual_texts,
        actual_types=actual_types,
    )


def test_extraction_gold_summary_is_clean() -> None:
    report = run_extraction_gold_evaluation()
    assert report.total_cases == len(EXTRACTION_GOLD_CASES)
    assert not report.failures, render_suite_report(report)


def test_structured_extraction_gold_summary_is_clean() -> None:
    report = run_extraction_gold_evaluation(mode="structured")

    assert report.total_cases == len(EXTRACTION_GOLD_CASES)
    assert not report.failures, render_suite_report(report)


def test_structured_extractor_matches_or_beats_legacy_on_harder_cases() -> None:
    harder_cases = tuple(
        case
        for case in EXTRACTION_GOLD_CASES
        if case.category in {"hybrid_fallback", "structured_extractor"}
    )

    legacy_report = run_extraction_gold_evaluation(harder_cases, mode="legacy")
    structured_report = run_extraction_gold_evaluation(harder_cases, mode="structured")

    assert structured_report.failed_cases <= legacy_report.failed_cases, (
        "Structured extractor regressed against the legacy baseline on harder gold cases.\n\n"
        f"Legacy:\n{render_suite_report(legacy_report)}\n\n"
        f"Structured:\n{render_suite_report(structured_report)}"
    )
