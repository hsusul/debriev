import uuid

from app.config import Settings
from app.core.provenance import build_block_anchor_metadata, build_exhibit_page_anchor_metadata, build_paragraph_anchor_metadata
from app.core.enums import ClaimType, LinkType, SupportStatus
from app.models.claim_unit import ClaimUnit
from app.models.segment import Segment
from app.models.support_link import SupportLink
from app.services.parsing.normalization import normalize_for_match
from app.services.llm.base import (
    ProviderRequest,
    ProviderResponse,
    ProviderSupportAssessment,
    VerificationProvider,
)
from app.services.verification.classifier import VerificationClassifier
from app.services.verification.evidence_roles import determine_evidence_role
from app.services.verification.heuristics import evaluate_heuristics


def build_claim(text: str) -> ClaimUnit:
    return ClaimUnit(
        assertion_id=uuid.uuid4(),
        text=text,
        normalized_text=normalize_for_match(text),
        claim_type=ClaimType.FACT,
        sequence_order=1,
        support_status=SupportStatus.UNVERIFIED,
    )


def build_segment(
    text: str,
    *,
    page_start: int | None = 10,
    line_start: int | None = 4,
    page_end: int | None = 10,
    line_end: int | None = 6,
    speaker: str = "A",
    segment_type: str = "ANSWER_BLOCK",
    anchor_metadata: dict[str, object] | None = None,
) -> Segment:
    return Segment(
        source_document_id=uuid.uuid4(),
        page_start=page_start,
        line_start=line_start,
        page_end=page_end,
        line_end=line_end,
        anchor_metadata=anchor_metadata,
        raw_text=text,
        normalized_text=normalize_for_match(text),
        speaker=speaker,
        segment_type=segment_type,
    )


def build_link() -> SupportLink:
    return SupportLink(
        claim_unit_id=uuid.uuid4(),
        segment_id=uuid.uuid4(),
        link_type=LinkType.MANUAL,
        citation_text="10:4-6",
        user_confirmed=True,
    )


class StubProvider(VerificationProvider):
    name = "stub"

    def __init__(self, response: ProviderResponse) -> None:
        self.response = response
        self.model_version = "stub-v1"
        self.calls = 0

    def verify(self, request: ProviderRequest) -> ProviderResponse:
        self.calls += 1
        return self.response


class StubClassifier(VerificationClassifier):
    def __init__(self, provider: StubProvider | None) -> None:
        super().__init__(settings=Settings())
        self.provider = provider

    def _build_provider(self, *, model_version: str):
        return self.provider


def test_verification_heuristics_flag_missing_citation() -> None:
    claim = build_claim("Doe signed the contract.")

    result = evaluate_heuristics(claim, links=[], segments=[])

    assert result.verdict == SupportStatus.UNSUPPORTED
    assert "missing_citation" in result.flags
    assert result.reasoning == ["No citation or verified authority supports this claim."]


def test_verification_heuristics_mark_case_citation_without_support_ambiguous() -> None:
    claim = build_claim("Under Smith v. Jones, Doe signed the contract.")

    result = evaluate_heuristics(claim, links=[], segments=[])

    assert result.verdict == SupportStatus.AMBIGUOUS
    assert "missing_citation" in result.flags
    assert result.reasoning == ["Citation present but no verified authority."]


def test_verification_heuristics_mark_strong_universal_without_support_overstated() -> None:
    claim = build_claim("Doe must always disclose all defects.")

    result = evaluate_heuristics(claim, links=[], segments=[])

    assert result.verdict == SupportStatus.OVERSTATED
    assert "missing_citation" in result.flags
    assert "absolute_qualifier_mismatch" in result.flags


def test_verification_heuristics_flag_absolute_qualifier_mismatch() -> None:
    claim = build_claim("Smith always approved invoices.")
    segment = build_segment("A. Smith approved the invoice on March 1.")
    link = build_link()

    result = evaluate_heuristics(claim, links=[link], segments=[segment])

    assert result.verdict == SupportStatus.OVERSTATED
    assert "absolute_qualifier_mismatch" in result.flags


def test_verification_heuristics_flag_subject_mismatch() -> None:
    claim = build_claim("Doe signed the contract.")
    segment = build_segment("A. Smith signed the contract.")
    link = build_link()

    result = evaluate_heuristics(claim, links=[link], segments=[segment])

    assert result.verdict == SupportStatus.AMBIGUOUS
    assert "subject_mismatch" in result.flags


def test_evidence_role_classification_is_source_agnostic() -> None:
    assert determine_evidence_role(segment_type="ANSWER_BLOCK", speaker="A", raw_text="A. Doe signed it.") == "substantive"
    assert determine_evidence_role(segment_type="QUESTION_BLOCK", speaker="Q", raw_text="Q. Did Doe sign it?") == "contextual"
    assert (
        determine_evidence_role(
            segment_type="DECLARATION_PARAGRAPH",
            speaker="DECLARANT ¶2",
            raw_text="2. I signed the declaration.",
        )
        == "substantive"
    )
    assert (
        determine_evidence_role(
            segment_type="DECLARATION_BLOCK",
            speaker="DECLARANT",
            raw_text="I, Jane Doe, declare as follows.",
        )
        == "contextual"
    )
    assert (
        determine_evidence_role(
            segment_type="DECLARATION_BLOCK",
            speaker="DECLARANT",
            raw_text="I signed the declaration and delivered it to counsel.",
        )
        == "substantive"
    )
    assert (
        determine_evidence_role(
            segment_type="EXHIBIT_PAGE_BLOCK",
            speaker=None,
            raw_text="The contract was signed on March 1.",
        )
        == "substantive"
    )
    assert (
        determine_evidence_role(
            segment_type="EXHIBIT_LABELED_BLOCK",
            speaker="Subject",
            raw_text="Subject: Contract Status",
        )
        == "contextual"
    )


def test_verification_heuristics_accept_declaration_paragraph_anchor_metadata() -> None:
    claim = build_claim("I signed the declaration.")
    segment = build_segment(
        "2. I signed the declaration.",
        page_start=None,
        line_start=None,
        page_end=None,
        line_end=None,
        speaker="DECLARANT ¶2",
        segment_type="DECLARATION_PARAGRAPH",
        anchor_metadata=build_paragraph_anchor_metadata(paragraph_number=2, sequence_order=2),
    )
    link = build_link()

    result = evaluate_heuristics(claim, links=[link], segments=[segment])

    assert "invalid_anchor" not in result.flags
    assert "contextual_support_only" not in result.flags
    assert result.verdict == SupportStatus.SUPPORTED


def test_verification_heuristics_treat_substantive_declaration_block_as_direct_support() -> None:
    claim = build_claim("I signed the declaration.")
    segment = build_segment(
        "I signed the declaration.",
        page_start=None,
        line_start=None,
        page_end=None,
        line_end=None,
        speaker="DECLARANT",
        segment_type="DECLARATION_BLOCK",
        anchor_metadata=build_block_anchor_metadata(block_index=2, sequence_order=2),
    )
    link = build_link()

    result = evaluate_heuristics(claim, links=[link], segments=[segment])

    assert "contextual_support_only" not in result.flags
    assert result.verdict == SupportStatus.SUPPORTED


def test_verification_heuristics_flag_temporal_scope_mismatch() -> None:
    claim = build_claim("Smith repeatedly approved invoices.")
    segment = build_segment("A. Smith approved the invoice on March 1.")
    link = build_link()

    result = evaluate_heuristics(claim, links=[link], segments=[segment])

    assert result.verdict == SupportStatus.OVERSTATED
    assert "temporal_scope_mismatch" in result.flags


def test_verification_heuristics_match_quoted_phrase_with_normalized_text() -> None:
    claim = build_claim('Smith testified, "I signed the agreement."')
    segment = build_segment("A. I signed the agreement.")
    link = build_link()

    result = evaluate_heuristics(claim, links=[link], segments=[segment])

    assert "quote_mismatch_placeholder" not in result.flags


def test_verification_heuristics_flag_knowledge_escalation() -> None:
    claim = build_claim("Doe knew the contract was fraudulent.")
    segment = build_segment("A. Doe signed the contract.")
    link = build_link()

    result = evaluate_heuristics(claim, links=[link], segments=[segment])

    assert result.verdict == SupportStatus.AMBIGUOUS
    assert "knowledge_escalation" in result.flags
    assert "needs_human_review" in result.flags


def test_verification_heuristics_flag_causation_escalation() -> None:
    claim = build_claim("Doe's conduct caused the spill.")
    segment = build_segment("A. Doe opened the valve and the spill happened later.")
    link = build_link()

    result = evaluate_heuristics(claim, links=[link], segments=[segment])

    assert result.verdict == SupportStatus.AMBIGUOUS
    assert "causation_escalation" in result.flags
    assert "needs_human_review" in result.flags


def test_verification_heuristics_flag_narrow_support_for_multi_predicate_claim() -> None:
    claim = build_claim("Doe reviewed the contract and approved the invoice.")
    segment = build_segment("A. Doe reviewed the contract.")
    link = build_link()

    result = evaluate_heuristics(claim, links=[link], segments=[segment])

    assert result.verdict == SupportStatus.OVERSTATED
    assert "narrow_support" in result.flags


def test_verification_heuristics_flag_narrow_support_for_single_supported_predicate() -> None:
    claim = build_claim("Doe reviewed the contract, approved the invoice, and emailed counsel.")
    segment = build_segment("A. Doe approved the invoice.")
    link = build_link()

    result = evaluate_heuristics(claim, links=[link], segments=[segment])

    assert result.verdict == SupportStatus.OVERSTATED
    assert "narrow_support" in result.flags


def test_verification_heuristics_flag_contextual_only_support() -> None:
    claim = build_claim("Doe signed the contract.")
    segment = build_segment(
        "Q. Did Doe sign the contract?",
        speaker="Q",
        segment_type="QUESTION_BLOCK",
    )
    link = build_link()

    result = evaluate_heuristics(claim, links=[link], segments=[segment])

    assert result.verdict == SupportStatus.AMBIGUOUS
    assert "contextual_support_only" in result.flags


def test_verification_heuristics_flag_contextual_only_support_for_declaration_boilerplate() -> None:
    claim = build_claim("Doe made the declaration under penalty of perjury.")
    segment = build_segment(
        "I declare under penalty of perjury that the foregoing is true and correct.",
        page_start=None,
        line_start=None,
        page_end=None,
        line_end=None,
        speaker="DECLARANT",
        segment_type="DECLARATION_BLOCK",
        anchor_metadata=build_block_anchor_metadata(block_index=3, sequence_order=3),
    )
    link = build_link()

    result = evaluate_heuristics(claim, links=[link], segments=[segment])

    assert result.verdict == SupportStatus.AMBIGUOUS
    assert "contextual_support_only" in result.flags


def test_verification_heuristics_accept_exhibit_page_anchor_metadata() -> None:
    claim = build_claim("The contract was signed on March 1.")
    segment = build_segment(
        "The contract was signed on March 1.",
        page_start=None,
        line_start=None,
        page_end=None,
        line_end=None,
        speaker=None,
        segment_type="EXHIBIT_PAGE_BLOCK",
        anchor_metadata=build_exhibit_page_anchor_metadata(page_number=1, block_index=2, sequence_order=2),
    )
    link = build_link()

    result = evaluate_heuristics(claim, links=[link], segments=[segment])

    assert "invalid_anchor" not in result.flags
    assert "contextual_support_only" not in result.flags
    assert result.verdict == SupportStatus.SUPPORTED


def test_verification_heuristics_flag_question_dominant_weak_answer_support() -> None:
    claim = build_claim("Doe signed the contract on March 1.")
    question_segment = build_segment(
        "Q. Did Doe sign the contract on March 1?",
        speaker="Q",
        segment_type="QUESTION_BLOCK",
    )
    weak_answer_segment = build_segment("A. I do not remember whether Doe signed it.", page_start=10, line_start=7)
    question_link = build_link()
    answer_link = build_link()

    result = evaluate_heuristics(
        claim,
        links=[question_link, answer_link],
        segments=[question_segment, weak_answer_segment],
    )

    assert result.verdict == SupportStatus.AMBIGUOUS
    assert "contextual_support_only" in result.flags
    assert "narrow_support" not in result.flags


def test_verification_heuristics_keep_all_weak_answers_cautious() -> None:
    claim = build_claim("Doe approved the invoice on March 1.")
    first_segment = build_segment("A. I do not remember whether Doe approved the invoice.")
    second_segment = build_segment("A. I am not sure if it happened on March 1.", page_start=10, line_start=7)
    first_link = build_link()
    second_link = build_link()

    result = evaluate_heuristics(
        claim,
        links=[first_link, second_link],
        segments=[first_segment, second_segment],
    )

    assert result.verdict == SupportStatus.AMBIGUOUS
    assert "equivocal" in " ".join(result.reasoning).lower()


def test_verification_classifier_keeps_blocking_flags_deterministic() -> None:
    claim = build_claim("Doe signed the contract.")
    provider = StubProvider(
        ProviderResponse(
            verdict=SupportStatus.SUPPORTED,
            reasoning="Provider would support this.",
            suggested_fix=None,
            confidence_score=0.9,
        )
    )
    classifier = StubClassifier(provider)

    result = classifier.verify(claim, links=[], segments=[])

    assert result.verdict == SupportStatus.UNSUPPORTED
    assert provider.calls == 0
    assert result.primary_anchor is None
    assert result.support_assessments == []


def test_verification_classifier_allows_warning_refinement() -> None:
    claim = build_claim("Smith always approved invoices.")
    segment = build_segment("A. Smith approved the invoice on March 1.")
    link = build_link()
    provider = StubProvider(
        ProviderResponse(
            verdict=SupportStatus.PARTIALLY_SUPPORTED,
            reasoning="The support is narrower than the claim.",
            suggested_fix="Narrow the assertion.",
            confidence_score=0.65,
            support_assessments=[
                ProviderSupportAssessment(
                    segment_id=segment.id,
                    anchor="10:4-6",
                    role="primary",
                    contribution="Main linked answer testimony.",
                )
            ],
            primary_anchor="10:4-6",
        )
    )
    classifier = StubClassifier(provider)

    result = classifier.verify(claim, links=[link], segments=[segment])

    assert result.verdict == SupportStatus.PARTIALLY_SUPPORTED
    assert provider.calls == 1
    assert result.suggested_fix == "Narrow the assertion."
    assert result.primary_anchor == "10:4-6"
    assert len(result.support_assessments) == 1


def test_verification_classifier_caps_contextual_only_provider_output() -> None:
    claim = build_claim("Doe signed the contract.")
    question_segment = build_segment(
        "Q. Did Doe sign the contract?",
        speaker="Q",
        segment_type="QUESTION_BLOCK",
    )
    link = build_link()
    provider = StubProvider(
        ProviderResponse(
            verdict=SupportStatus.SUPPORTED,
            reasoning="Provider treated the question as support.",
            suggested_fix=None,
            confidence_score=0.75,
            support_assessments=[
                ProviderSupportAssessment(
                    segment_id=question_segment.id,
                    anchor="10:4-6",
                    role="contextual",
                    contribution="Only question framing overlaps the claim.",
                )
            ],
            primary_anchor="10:4-6",
        )
    )
    classifier = StubClassifier(provider)

    result = classifier.verify(claim, links=[link], segments=[question_segment])

    assert provider.calls == 1
    assert result.verdict == SupportStatus.AMBIGUOUS


def test_verification_classifier_caps_escalation_provider_output() -> None:
    claim = build_claim("Doe knew the contract was fraudulent.")
    segment = build_segment("A. Doe signed the contract.")
    link = build_link()
    provider = StubProvider(
        ProviderResponse(
            verdict=SupportStatus.PARTIALLY_SUPPORTED,
            reasoning="Provider found some overlap.",
            suggested_fix=None,
            confidence_score=0.6,
            support_assessments=[
                ProviderSupportAssessment(
                    segment_id=segment.id,
                    anchor="10:4-6",
                    role="primary",
                    contribution="Main linked answer testimony.",
                )
            ],
            primary_anchor="10:4-6",
        )
    )
    classifier = StubClassifier(provider)

    result = classifier.verify(claim, links=[link], segments=[segment])

    assert provider.calls == 1
    assert result.verdict == SupportStatus.AMBIGUOUS
