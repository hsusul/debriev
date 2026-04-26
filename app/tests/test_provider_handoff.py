import uuid

from app.config import Settings
from app.core.enums import ClaimType, LinkType, SupportStatus
from app.models.claim_unit import ClaimUnit
from app.models.segment import Segment
from app.models.support_link import SupportLink
from app.services.llm.base import (
    ProviderRequest,
    ProviderResponse,
    ProviderSupportAssessment,
    ProviderSupportItem,
    VerificationProvider,
)
from app.services.llm.openai_provider import OpenAIProvider
from app.services.parsing.normalization import normalize_for_match
from app.services.verification.classifier import VerificationClassifier
from app.services.verification.evidence_roles import determine_evidence_role


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
    page_start: int,
    line_start: int,
    page_end: int,
    line_end: int,
    speaker: str = "A",
    segment_type: str = "ANSWER_BLOCK",
) -> Segment:
    return Segment(
        id=uuid.uuid4(),
        source_document_id=uuid.uuid4(),
        page_start=page_start,
        line_start=line_start,
        page_end=page_end,
        line_end=line_end,
        raw_text=text,
        normalized_text=normalize_for_match(text),
        speaker=speaker,
        segment_type=segment_type,
    )


def build_link(
    segment: Segment,
    *,
    sequence_order: int,
    citation_text: str | None = None,
) -> SupportLink:
    return SupportLink(
        id=uuid.uuid4(),
        claim_unit_id=uuid.uuid4(),
        segment_id=segment.id,
        sequence_order=sequence_order,
        link_type=LinkType.MANUAL,
        citation_text=citation_text,
        user_confirmed=True,
        segment=segment,
    )


class CapturingProvider(VerificationProvider):
    name = "capturing"

    def __init__(self) -> None:
        self.model_version = "capturing-v1"
        self.last_request: ProviderRequest | None = None

    def verify(self, request: ProviderRequest) -> ProviderResponse:
        self.last_request = request
        return ProviderResponse(
            verdict=SupportStatus.PARTIALLY_SUPPORTED,
            reasoning="Structured support bundle inspected.",
            suggested_fix=None,
            confidence_score=0.6,
            support_assessments=[
                ProviderSupportAssessment(
                    segment_id=item.segment_id,
                    anchor=item.anchor,
                    role="primary" if index == 0 else "contextual",
                    contribution="Captured in test provider.",
                )
                for index, item in enumerate(request.support_items)
            ],
            primary_anchor=request.support_items[0].anchor if request.support_items else None,
        )


class CapturingClassifier(VerificationClassifier):
    def __init__(self, provider: CapturingProvider) -> None:
        super().__init__(settings=Settings())
        self.provider = provider

    def _build_provider(self, *, model_version: str):
        return self.provider


def test_provider_request_gets_structured_support_items_in_order() -> None:
    claim = build_claim("Smith signed the contract.")
    first_segment = build_segment(
        "Q. Did you review the contract?",
        page_start=10,
        line_start=1,
        page_end=10,
        line_end=2,
        speaker="Q",
        segment_type="QUESTION_BLOCK",
    )
    second_segment = build_segment(
        "A. Smith signed it.",
        page_start=10,
        line_start=3,
        page_end=10,
        line_end=4,
    )
    links = [
        build_link(second_segment, sequence_order=1),
        build_link(first_segment, sequence_order=2),
    ]
    provider = CapturingProvider()
    classifier = CapturingClassifier(provider)

    result = classifier.verify(claim, links=links, segments=[first_segment, second_segment])

    assert result.verdict == SupportStatus.PARTIALLY_SUPPORTED
    assert provider.last_request is not None
    assert [item.segment_id for item in provider.last_request.support_items] == [
        second_segment.id,
        first_segment.id,
    ]
    assert provider.last_request.support_items[0].anchor == "p.10:3-10:4"
    assert [item.evidence_role for item in provider.last_request.support_items] == [
        "substantive",
        "contextual",
    ]
    assert provider.last_request.support_items[0].speaker == "A"
    assert provider.last_request.support_items[1].segment_type == "QUESTION_BLOCK"


def test_provider_request_retains_context_citations_and_heuristic_flags() -> None:
    claim = build_claim("Smith always approved invoices.")
    segment = build_segment(
        "A. Smith approved the invoice on March 1.",
        page_start=10,
        line_start=4,
        page_end=10,
        line_end=6,
    )
    link = build_link(segment, sequence_order=1, citation_text="10:4-6")
    provider = CapturingProvider()
    classifier = CapturingClassifier(provider)

    classifier.verify(claim, links=[link], segments=[segment])

    assert provider.last_request is not None
    assert provider.last_request.citations == ["10:4-6"]
    assert "absolute_qualifier_mismatch" in provider.last_request.heuristic_flags
    assert "Anchor: 10:4-6" in provider.last_request.context
    assert "A. Smith approved the invoice on March 1." in provider.last_request.context


def test_multi_support_order_is_preserved_through_classifier_handoff() -> None:
    claim = build_claim("Doe reviewed the contract and signed it because payment was due.")
    first_segment = build_segment(
        "A. I reviewed the contract.",
        page_start=12,
        line_start=7,
        page_end=12,
        line_end=8,
    )
    second_segment = build_segment(
        "A. I signed it later that day.",
        page_start=12,
        line_start=9,
        page_end=12,
        line_end=10,
    )
    third_segment = build_segment(
        "A. Payment was due that day.",
        page_start=12,
        line_start=11,
        page_end=12,
        line_end=12,
    )
    links = [
        build_link(first_segment, sequence_order=2),
        build_link(third_segment, sequence_order=3),
        build_link(second_segment, sequence_order=1),
    ]
    provider = CapturingProvider()
    classifier = CapturingClassifier(provider)

    classifier.verify(claim, links=links, segments=[third_segment, first_segment, second_segment])

    assert provider.last_request is not None
    assert [item.anchor for item in provider.last_request.support_items] == [
        "p.12:9-12:10",
        "p.12:7-12:8",
        "p.12:11-12:12",
    ]
    assert provider.last_request.citations == [
        "p.12:9-12:10",
        "p.12:7-12:8",
        "p.12:11-12:12",
    ]


def test_placeholder_provider_returns_primary_and_secondary_support_assessments() -> None:
    provider = OpenAIProvider(api_key="test", model_version="stub-v1")
    strong_segment = build_segment(
        "A. Smith signed the contract on March 1.",
        page_start=10,
        line_start=3,
        page_end=10,
        line_end=4,
    )
    weak_segment = build_segment(
        "Q. Did you review the file before the meeting?",
        page_start=10,
        line_start=1,
        page_end=10,
        line_end=2,
        speaker="Q",
        segment_type="QUESTION_BLOCK",
    )
    request = ProviderRequest(
        claim_text="Smith signed the contract.",
        support_items=[
            build_provider_support_item(weak_segment),
            build_provider_support_item(strong_segment),
        ],
        context="placeholder",
        citations=["p.10:1-10:2", "p.10:3-10:4"],
        heuristic_flags=[],
    )

    response = provider.verify(request)

    assert response.primary_anchor == "p.10:3-10:4"
    assert len(response.support_assessments) == 2
    assert [assessment.anchor for assessment in response.support_assessments] == [
        "p.10:1-10:2",
        "p.10:3-10:4",
    ]
    assert response.support_assessments[0].role == "contextual"
    assert response.support_assessments[1].role == "primary"


def test_placeholder_provider_prefers_answer_block_over_question_block() -> None:
    provider = OpenAIProvider(api_key="test", model_version="stub-v1")
    question_segment = build_segment(
        "Q. Did Smith sign the contract?",
        page_start=15,
        line_start=1,
        page_end=15,
        line_end=2,
        speaker="Q",
        segment_type="QUESTION_BLOCK",
    )
    answer_segment = build_segment(
        "A. Yes, Smith signed the contract.",
        page_start=15,
        line_start=3,
        page_end=15,
        line_end=4,
        speaker="A",
        segment_type="ANSWER_BLOCK",
    )
    request = ProviderRequest(
        claim_text="Smith signed the contract.",
        support_items=[
            build_provider_support_item(question_segment),
            build_provider_support_item(answer_segment),
        ],
        context="placeholder",
        citations=["p.15:1-15:2", "p.15:3-15:4"],
        heuristic_flags=[],
    )

    response = provider.verify(request)

    assert response.primary_anchor == "p.15:3-15:4"
    assert response.support_assessments[0].role in {"secondary", "contextual"}
    assert response.support_assessments[1].role == "primary"


def test_placeholder_provider_uses_declaration_roles_for_primary_support() -> None:
    provider = OpenAIProvider(api_key="test", model_version="stub-v1")
    request = ProviderRequest(
        claim_text="I signed the declaration.",
        support_items=[
            ProviderSupportItem(
                segment_id=uuid.uuid4(),
                anchor="block 1",
                evidence_role="contextual",
                speaker="DECLARANT",
                segment_type="DECLARATION_BLOCK",
                raw_text="I, Jane Doe, declare as follows.",
                normalized_text=normalize_for_match("I, Jane Doe, declare as follows."),
            ),
            ProviderSupportItem(
                segment_id=uuid.uuid4(),
                anchor="¶2",
                evidence_role="substantive",
                speaker="DECLARANT ¶2",
                segment_type="DECLARATION_PARAGRAPH",
                raw_text="2. I signed the declaration.",
                normalized_text=normalize_for_match("2. I signed the declaration."),
            ),
        ],
        context="placeholder",
        citations=["block 1", "¶2"],
        heuristic_flags=[],
    )

    response = provider.verify(request)

    assert response.primary_anchor == "¶2"
    assert [assessment.role for assessment in response.support_assessments] == [
        "contextual",
        "primary",
    ]


def test_placeholder_provider_keeps_contextual_declaration_block_cautious() -> None:
    provider = OpenAIProvider(api_key="test", model_version="stub-v1")
    request = ProviderRequest(
        claim_text="Doe made the declaration under penalty of perjury.",
        support_items=[
            ProviderSupportItem(
                segment_id=uuid.uuid4(),
                anchor="block 3",
                evidence_role="contextual",
                speaker="DECLARANT",
                segment_type="DECLARATION_BLOCK",
                raw_text="I declare under penalty of perjury that the foregoing is true and correct.",
                normalized_text=normalize_for_match(
                    "I declare under penalty of perjury that the foregoing is true and correct."
                ),
            ),
        ],
        context="placeholder",
        citations=["block 3"],
        heuristic_flags=[],
    )

    response = provider.verify(request)

    assert response.verdict == SupportStatus.AMBIGUOUS
    assert response.primary_anchor == "block 3"
    assert response.support_assessments[0].role == "contextual"


def test_placeholder_provider_keeps_contextual_only_support_cautious() -> None:
    provider = OpenAIProvider(api_key="test", model_version="stub-v1")
    question_segment = build_segment(
        "Q. Did Smith sign the contract?",
        page_start=16,
        line_start=1,
        page_end=16,
        line_end=2,
        speaker="Q",
        segment_type="QUESTION_BLOCK",
    )
    request = ProviderRequest(
        claim_text="Smith signed the contract.",
        support_items=[build_provider_support_item(question_segment)],
        context="placeholder",
        citations=["p.16:1-16:2"],
        heuristic_flags=[],
    )

    response = provider.verify(request)

    assert response.verdict == SupportStatus.AMBIGUOUS
    assert response.primary_anchor == "p.16:1-16:2"
    assert response.support_assessments[0].role == "contextual"


def test_placeholder_provider_marks_distributed_support_as_partial() -> None:
    provider = OpenAIProvider(api_key="test", model_version="stub-v1")
    first_segment = build_segment(
        "A. Doe reviewed the contract.",
        page_start=17,
        line_start=1,
        page_end=17,
        line_end=2,
    )
    second_segment = build_segment(
        "A. Doe approved the invoice.",
        page_start=17,
        line_start=3,
        page_end=17,
        line_end=4,
    )
    request = ProviderRequest(
        claim_text="Doe reviewed the contract and approved the invoice.",
        support_items=[
            build_provider_support_item(first_segment),
            build_provider_support_item(second_segment),
        ],
        context="placeholder",
        citations=["p.17:1-17:2", "p.17:3-17:4"],
        heuristic_flags=[],
    )

    response = provider.verify(request)

    assert response.verdict == SupportStatus.PARTIALLY_SUPPORTED
    assert [assessment.role for assessment in response.support_assessments] == [
        "primary",
        "secondary",
    ]
    assert any("narrow slice of the claim" in assessment.contribution for assessment in response.support_assessments)


def test_placeholder_provider_stays_cautious_when_question_outweighs_answer() -> None:
    provider = OpenAIProvider(api_key="test", model_version="stub-v1")
    question_segment = build_segment(
        "Q. Did Smith sign the contract on March 1?",
        page_start=18,
        line_start=1,
        page_end=18,
        line_end=2,
        speaker="Q",
        segment_type="QUESTION_BLOCK",
    )
    answer_segment = build_segment(
        "A. I do not remember.",
        page_start=18,
        line_start=3,
        page_end=18,
        line_end=4,
        speaker="A",
        segment_type="ANSWER_BLOCK",
    )
    request = ProviderRequest(
        claim_text="Smith signed the contract on March 1.",
        support_items=[
            build_provider_support_item(question_segment),
            build_provider_support_item(answer_segment),
        ],
        context="placeholder",
        citations=["p.18:1-18:2", "p.18:3-18:4"],
        heuristic_flags=[],
    )

    response = provider.verify(request)

    assert response.verdict == SupportStatus.AMBIGUOUS
    assert response.primary_anchor == "p.18:1-18:2"
    assert [assessment.role for assessment in response.support_assessments] == [
        "contextual",
        "contextual",
    ]
    assert "contextual" in response.reasoning.lower()
    assert "substantive" in response.reasoning.lower()
    assert "support" in response.reasoning.lower()


def test_placeholder_provider_keeps_all_weak_answers_ambiguous() -> None:
    provider = OpenAIProvider(api_key="test", model_version="stub-v1")
    first_segment = build_segment(
        "A. I do not remember whether Doe approved the invoice.",
        page_start=21,
        line_start=1,
        page_end=21,
        line_end=2,
    )
    second_segment = build_segment(
        "A. I am not sure if it happened on March 1.",
        page_start=21,
        line_start=3,
        page_end=21,
        line_end=4,
    )
    request = ProviderRequest(
        claim_text="Doe approved the invoice on March 1.",
        support_items=[
            build_provider_support_item(first_segment),
            build_provider_support_item(second_segment),
        ],
        context="placeholder",
        citations=["p.21:1-21:2", "p.21:3-21:4"],
        heuristic_flags=[],
    )

    response = provider.verify(request)

    assert response.verdict == SupportStatus.AMBIGUOUS
    assert [assessment.role for assessment in response.support_assessments] == [
        "contextual",
        "contextual",
    ]
    assert "equivocal" in response.reasoning.lower()


def test_placeholder_provider_downgrades_mixed_direct_and_equivocal_answers() -> None:
    provider = OpenAIProvider(api_key="test", model_version="stub-v1")
    strong_segment = build_segment(
        "A. Doe signed the contract.",
        page_start=22,
        line_start=1,
        page_end=22,
        line_end=2,
    )
    weak_segment = build_segment(
        "A. I do not recall whether Doe signed the contract.",
        page_start=22,
        line_start=3,
        page_end=22,
        line_end=4,
    )
    request = ProviderRequest(
        claim_text="Doe signed the contract.",
        support_items=[
            build_provider_support_item(strong_segment),
            build_provider_support_item(weak_segment),
        ],
        context="placeholder",
        citations=["p.22:1-22:2", "p.22:3-22:4"],
        heuristic_flags=[],
    )

    response = provider.verify(request)

    assert response.verdict == SupportStatus.PARTIALLY_SUPPORTED
    assert response.primary_anchor == "p.22:1-22:2"
    assert [assessment.role for assessment in response.support_assessments] == [
        "primary",
        "contextual",
    ]
    assert "equivocal" in response.reasoning.lower()


def test_placeholder_provider_marks_distributed_three_item_support_as_partial() -> None:
    provider = OpenAIProvider(api_key="test", model_version="stub-v1")
    first_segment = build_segment(
        "A. Doe reviewed the contract.",
        page_start=19,
        line_start=1,
        page_end=19,
        line_end=2,
    )
    second_segment = build_segment(
        "A. Doe approved the invoice.",
        page_start=19,
        line_start=3,
        page_end=19,
        line_end=4,
    )
    third_segment = build_segment(
        "A. Doe emailed counsel.",
        page_start=19,
        line_start=5,
        page_end=19,
        line_end=6,
    )
    request = ProviderRequest(
        claim_text="Doe reviewed the contract, approved the invoice, and emailed counsel.",
        support_items=[
            build_provider_support_item(first_segment),
            build_provider_support_item(second_segment),
            build_provider_support_item(third_segment),
        ],
        context="placeholder",
        citations=["p.19:1-19:2", "p.19:3-19:4", "p.19:5-19:6"],
        heuristic_flags=[],
    )

    response = provider.verify(request)

    assert response.verdict == SupportStatus.PARTIALLY_SUPPORTED
    assert response.primary_anchor == "p.19:1-19:2"
    assert [assessment.role for assessment in response.support_assessments] == [
        "primary",
        "secondary",
        "secondary",
    ]


def test_classifier_reasoning_includes_primary_anchor_and_support_summary() -> None:
    claim = build_claim("Smith signed the contract.")
    strong_segment = build_segment(
        "A. Smith signed the contract on March 1.",
        page_start=20,
        line_start=5,
        page_end=20,
        line_end=6,
    )
    weak_segment = build_segment(
        "Q. Did you attend the meeting?",
        page_start=20,
        line_start=1,
        page_end=20,
        line_end=2,
        speaker="Q",
        segment_type="QUESTION_BLOCK",
    )
    links = [
        build_link(weak_segment, sequence_order=1),
        build_link(strong_segment, sequence_order=2),
    ]
    provider = OpenAIProvider(api_key="test", model_version="stub-v1")
    classifier = CapturingClassifier(provider)

    result = classifier.verify(claim, links=links, segments=[strong_segment, weak_segment])

    assert "Primary support anchor: p.20:5-20:6." in result.reasoning
    assert "Structured support reasoning:" in result.reasoning
    assert "p.20:1-20:2 [contextual]" in result.reasoning
    assert "p.20:5-20:6 [primary]" in result.reasoning


def test_classifier_result_exposes_structured_provider_output() -> None:
    claim = build_claim("Smith signed the contract.")
    strong_segment = build_segment(
        "A. Smith signed the contract on March 1.",
        page_start=30,
        line_start=5,
        page_end=30,
        line_end=6,
    )
    weak_segment = build_segment(
        "Q. Did you attend the meeting?",
        page_start=30,
        line_start=1,
        page_end=30,
        line_end=2,
        speaker="Q",
        segment_type="QUESTION_BLOCK",
    )
    links = [
        build_link(weak_segment, sequence_order=1),
        build_link(strong_segment, sequence_order=2),
    ]
    provider = OpenAIProvider(api_key="test", model_version="stub-v1")
    classifier = CapturingClassifier(provider)

    result = classifier.verify(claim, links=links, segments=[strong_segment, weak_segment])

    assert result.primary_anchor == "p.30:5-30:6"
    assert [assessment.anchor for assessment in result.support_assessments] == [
        "p.30:1-30:2",
        "p.30:5-30:6",
    ]
    assert result.support_assessments[0].role == "contextual"
    assert result.support_assessments[1].role == "primary"


def test_classifier_result_deterministic_only_has_empty_structured_output() -> None:
    claim = build_claim("Smith signed the contract.")
    segment = build_segment(
        "A. Smith signed the contract on March 1.",
        page_start=40,
        line_start=3,
        page_end=40,
        line_end=4,
    )
    link = build_link(segment, sequence_order=1)
    classifier = CapturingClassifier(None)

    result = classifier.verify(claim, links=[link], segments=[segment])

    assert result.primary_anchor is None
    assert result.support_assessments == []
    assert "No LLM provider configured" in result.reasoning


def build_provider_support_item(segment: Segment):
    from app.services.llm.base import ProviderSupportItem

    return ProviderSupportItem(
        segment_id=segment.id,
        anchor=f"p.{segment.page_start}:{segment.line_start}-{segment.page_end}:{segment.line_end}",
        evidence_role=determine_evidence_role(
            segment_type=segment.segment_type,
            speaker=segment.speaker,
            raw_text=segment.raw_text,
        ),
        speaker=segment.speaker,
        segment_type=segment.segment_type,
        raw_text=segment.raw_text,
        normalized_text=segment.normalized_text,
    )
