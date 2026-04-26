from app.core.enums import ClaimType
from app.services.claims.extractor import (
    extract_claim_candidates,
    extract_legacy_claim_candidates,
    extract_structured_claim_candidates,
    extract_claim_candidates_with_strategy,
    select_extraction_strategy,
)


def test_claim_extraction_splits_obvious_conjunctions() -> None:
    text = "Doe signed the contract and approved the invoice because payment was due."

    claims = extract_claim_candidates(text)

    assert len(claims) == 2
    assert claims[0].claim_type == ClaimType.FACT
    assert claims[1].claim_type == ClaimType.INFERENCE
    assert claims[0].sequence_order == 1
    assert claims[1].sequence_order == 2


def test_claim_extraction_splits_shared_subject_distinct_predicates() -> None:
    claims = extract_claim_candidates("Doe signed the contract and approved the invoice.")

    assert len(claims) == 2
    assert claims[0].text == "Doe signed the contract"
    assert claims[1].text == "Doe approved the invoice."
    assert all(claim.claim_type == ClaimType.FACT for claim in claims)


def test_claim_extraction_splits_comma_separated_predicate_list() -> None:
    claims = extract_claim_candidates("Doe reviewed the contract, signed the declaration, and emailed counsel.")

    assert [claim.text for claim in claims] == [
        "Doe reviewed the contract",
        "Doe signed the declaration",
        "Doe emailed counsel.",
    ]


def test_claim_extraction_does_not_treat_apostrophes_as_quotes() -> None:
    claims = extract_claim_candidates("Doe's explanation wasn't consistent with the exhibit.")

    assert len(claims) == 1
    assert claims[0].claim_type == ClaimType.FACT


def test_claim_extraction_detects_double_quoted_language() -> None:
    claims = extract_claim_candidates('Doe testified, "I signed the agreement."')

    assert len(claims) == 1
    assert claims[0].claim_type == ClaimType.QUOTE


def test_claim_extraction_splits_quote_from_inferential_gloss() -> None:
    claims = extract_claim_candidates('Doe testified, "I signed the agreement," which suggests he knew the terms.')

    assert len(claims) == 2
    assert claims[0].claim_type == ClaimType.QUOTE
    assert claims[1].claim_type == ClaimType.INFERENCE
    assert claims[1].text == "which suggests he knew the terms."


def test_claim_extraction_keeps_embedded_quote_attached_without_inferential_gloss() -> None:
    claims = extract_claim_candidates('Counsel noted that Doe testified, "I reviewed the file and signed the declaration."')

    assert len(claims) == 1
    assert claims[0].claim_type == ClaimType.QUOTE


def test_claim_extraction_marks_fact_plus_inferential_gloss_as_mixed() -> None:
    claims = extract_claim_candidates("Doe signed the contract, which suggests he knew the terms.")

    assert len(claims) == 1
    assert claims[0].claim_type == ClaimType.MIXED


def test_claim_extraction_marks_suggesting_gloss_as_mixed() -> None:
    claims = extract_claim_candidates("Doe signed the release, suggesting he understood the waiver.")

    assert len(claims) == 1
    assert claims[0].claim_type == ClaimType.MIXED


def test_claim_extraction_keeps_unsafe_compound_sentence_as_one_claim() -> None:
    claims = extract_claim_candidates("Doe reviewed and approved the agreement.")

    assert len(claims) == 1
    assert claims[0].text == "Doe reviewed and approved the agreement."


def test_claim_extraction_splits_later_predicate_after_noun_phrase_conjunction() -> None:
    claims = extract_claim_candidates("Doe reviewed the purchase and sale agreement and signed the affidavit.")

    assert [claim.text for claim in claims] == [
        "Doe reviewed the purchase and sale agreement",
        "Doe signed the affidavit.",
    ]


def test_claim_extraction_keeps_shared_object_list_as_one_claim() -> None:
    claims = extract_claim_candidates("Doe met with Smith and Jones at the office.")

    assert len(claims) == 1
    assert claims[0].text == "Doe met with Smith and Jones at the office."


def test_claim_extraction_splits_later_follow_on_predicate() -> None:
    claims = extract_claim_candidates("Doe reviewed the agreement and later signed the declaration.")

    assert [claim.text for claim in claims] == [
        "Doe reviewed the agreement",
        "Doe later signed the declaration.",
    ]


def test_claim_extraction_splits_quote_from_later_fact_clause() -> None:
    claims = extract_claim_candidates('Doe testified, "I signed the agreement," and later emailed counsel about the closing.')

    assert [claim.text for claim in claims] == [
        'Doe testified, "I signed the agreement,"',
        "Doe later emailed counsel about the closing.",
    ]
    assert [claim.claim_type for claim in claims] == [ClaimType.QUOTE, ClaimType.FACT]


def test_claim_extraction_keeps_noun_phrase_coordination_before_later_predicate() -> None:
    claims = extract_claim_candidates(
        "Doe reviewed the purchase and sale agreement and related schedules and later signed the affidavit."
    )

    assert [claim.text for claim in claims] == [
        "Doe reviewed the purchase and sale agreement and related schedules",
        "Doe later signed the affidavit.",
    ]


def test_claim_extraction_splits_shared_subject_chain_after_noun_phrase_coordination() -> None:
    claims = extract_claim_candidates("Doe reviewed the agreement and related schedules and circulated the revised declaration.")

    assert [claim.text for claim in claims] == [
        "Doe reviewed the agreement and related schedules",
        "Doe circulated the revised declaration.",
    ]


def test_claim_extraction_splits_embedded_quote_from_inferential_commentary() -> None:
    claims = extract_claim_candidates(
        'Counsel argued that Doe testified, "I reviewed the file," suggesting he knew the discrepancy.'
    )

    assert [claim.text for claim in claims] == [
        'Counsel argued that Doe testified, "I reviewed the file,"',
        "suggesting he knew the discrepancy.",
    ]
    assert [claim.claim_type for claim in claims] == [ClaimType.QUOTE, ClaimType.INFERENCE]


def test_claim_extraction_keeps_comma_separated_object_list_intact() -> None:
    claims = extract_claim_candidates("Doe reviewed the agreement, related schedules, and draft disclosures.")

    assert len(claims) == 1
    assert claims[0].text == "Doe reviewed the agreement, related schedules, and draft disclosures."


def test_claim_extraction_splits_temporal_follow_on_predicate_chain() -> None:
    claims = extract_claim_candidates("Doe reviewed the agreement, then signed the declaration and emailed counsel.")

    assert [claim.text for claim in claims] == [
        "Doe reviewed the agreement",
        "Doe then signed the declaration",
        "Doe emailed counsel.",
    ]


def test_claim_extraction_selector_keeps_easy_case_on_heuristic_path() -> None:
    assert select_extraction_strategy("Doe signed the contract and approved the invoice.") == "legacy"


def test_claim_extraction_selector_uses_fallback_for_exhibit_style_preamble() -> None:
    text = "Subject: Contract Status. Doe signed the agreement and later emailed counsel regarding payment."

    result = extract_claim_candidates_with_strategy(text)

    assert result.strategy == "structured"
    assert [claim.text for claim in result.claims] == [
        "Doe signed the agreement",
        "Doe later emailed counsel regarding payment.",
    ]


def test_claim_extraction_fallback_handles_reported_exhibit_predicate_chain() -> None:
    text = (
        "Counsel contended that the exhibit reflects Doe signed the contract, approved the invoice, "
        "and thereafter forwarded the executed copy to counsel."
    )

    result = extract_claim_candidates_with_strategy(text)

    assert result.strategy == "structured"
    assert [claim.text for claim in result.claims] == [
        "Counsel contended that the exhibit reflects Doe signed the contract",
        "Doe approved the invoice",
        "Doe thereafter forwarded the executed copy to counsel.",
    ]
    assert [claim.claim_type for claim in result.claims] == [
        ClaimType.FACT,
        ClaimType.FACT,
        ClaimType.FACT,
    ]


def test_claim_extraction_fallback_splits_quote_gloss_and_follow_on_fact() -> None:
    text = (
        'Counsel argued that Doe testified, "I reviewed the file and signed the agreement," '
        "which suggests he knew the discrepancy, and later moved to compel production."
    )

    result = extract_claim_candidates_with_strategy(text)

    assert result.strategy == "structured"
    assert [claim.text for claim in result.claims] == [
        'Counsel argued that Doe testified, "I reviewed the file and signed the agreement,"',
        "which suggests he knew the discrepancy",
        "Counsel later moved to compel production.",
    ]
    assert [claim.claim_type for claim in result.claims] == [
        ClaimType.QUOTE,
        ClaimType.INFERENCE,
        ClaimType.FACT,
    ]


def test_structured_extractor_handles_exhibit_header_quote_and_follow_on_fact() -> None:
    text = 'From: Jane Doe. In the March 1 email, Doe wrote, "I signed the release," and later sent the executed copy to counsel.'

    claims = extract_structured_claim_candidates(text)

    assert [claim.text for claim in claims] == [
        'In the March 1 email, Doe wrote, "I signed the release,"',
        "Doe later sent the executed copy to counsel.",
    ]
    assert [claim.claim_type for claim in claims] == [ClaimType.QUOTE, ClaimType.FACT]


def test_structured_extractor_handles_nested_reporting_chain() -> None:
    text = (
        "Counsel argued that the exhibit indicates Doe reviewed the agreement, approved the invoice, "
        "and later forwarded the executed copy."
    )

    claims = extract_structured_claim_candidates(text)

    assert [claim.text for claim in claims] == [
        "Counsel argued that the exhibit indicates Doe reviewed the agreement",
        "Doe approved the invoice",
        "Doe later forwarded the executed copy.",
    ]


def test_structured_extractor_handles_negated_clause_with_follow_on_fact() -> None:
    text = "Doe signed the agreement but not the guaranty, and later emailed counsel about the exception."

    claims = extract_structured_claim_candidates(text)

    assert [claim.text for claim in claims] == [
        "Doe signed the agreement but not the guaranty",
        "Doe later emailed counsel about the exception.",
    ]


def test_structured_extractor_beats_legacy_on_exhibit_style_sentence() -> None:
    text = 'From: Jane Doe. In the March 1 email, Doe wrote, "I signed the release," and later sent the executed copy to counsel.'

    legacy_claims = extract_legacy_claim_candidates(text)
    structured_claims = extract_structured_claim_candidates(text)

    assert len(legacy_claims) < len(structured_claims)
    assert [claim.text for claim in structured_claims] == [
        'In the March 1 email, Doe wrote, "I signed the release,"',
        "Doe later sent the executed copy to counsel.",
    ]


def test_structured_extractor_matches_gold_on_nested_reporting_chain() -> None:
    text = (
        "Counsel argued that the exhibit indicates Doe reviewed the agreement, approved the invoice, "
        "and later forwarded the executed copy."
    )

    structured_claims = extract_structured_claim_candidates(text)

    assert [claim.text for claim in structured_claims] == [
        "Counsel argued that the exhibit indicates Doe reviewed the agreement",
        "Doe approved the invoice",
        "Doe later forwarded the executed copy.",
    ]
