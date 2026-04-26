from collections.abc import Sequence
import uuid

import pytest

from app.core.enums import ClaimType, LinkType, SupportStatus
from app.models.claim_unit import ClaimUnit
from app.models.segment import Segment
from app.models.support_link import SupportLink
from app.services.llm.base import ProviderRequest, ProviderSupportItem
from app.services.llm.openai_provider import OpenAIProvider
from app.services.parsing.normalization import normalize_for_match
from app.services.verification.evidence_roles import determine_evidence_role
from app.services.verification.heuristics import evaluate_heuristics
from app.tests.gold_eval_report import (
    render_heuristic_mismatch,
    render_provider_mismatch,
    render_suite_report,
    run_verification_heuristic_gold_evaluation,
    run_verification_provider_gold_evaluation,
)
from app.tests.gold_sets import VerificationGoldCase, VerificationSupportItemGold, VERIFICATION_GOLD_CASES


@pytest.mark.parametrize("case", VERIFICATION_GOLD_CASES, ids=lambda case: case.name)
def test_verification_heuristic_gold_set(case: VerificationGoldCase) -> None:
    claim = _build_claim(case.claim_text)
    segments = [_build_segment(item) for item in case.support_items]
    links = [_build_link(segment) for segment in segments]

    result = evaluate_heuristics(claim, links=links, segments=segments)
    actual_flags = tuple(sorted(result.flags))
    expected_flags = tuple(sorted(case.expected_flags))

    assert actual_flags == expected_flags, render_heuristic_mismatch(
        case,
        actual_flags=actual_flags,
        actual_verdict=result.verdict,
    )
    assert result.verdict == case.expected_verdict, render_heuristic_mismatch(
        case,
        actual_flags=actual_flags,
        actual_verdict=result.verdict,
    )


@pytest.mark.parametrize(
    "case",
    [case for case in VERIFICATION_GOLD_CASES if case.expected_provider_verdict is not None],
    ids=lambda case: case.name,
)
def test_verification_provider_gold_set(case: VerificationGoldCase) -> None:
    provider = OpenAIProvider(api_key="test", model_version="eval-v1")
    support_items = [_build_provider_support_item(item) for item in case.support_items]
    response = provider.verify(
        ProviderRequest(
            claim_text=case.claim_text,
            support_items=support_items,
            context=_render_context_bundle(case.support_items),
            citations=[item.anchor for item in case.support_items if item.anchor is not None],
            heuristic_flags=list(case.expected_flags),
        )
    )

    actual_roles = tuple(assessment.role for assessment in response.support_assessments)
    actual_anchors = tuple(assessment.anchor for assessment in response.support_assessments)
    expected_anchors = tuple(item.anchor for item in case.support_items if item.anchor is not None)

    assert response.verdict == case.expected_provider_verdict, render_provider_mismatch(
        case,
        actual_verdict=response.verdict,
        actual_primary_anchor=response.primary_anchor,
        actual_roles=actual_roles,
        actual_anchors=actual_anchors,
    )
    assert response.primary_anchor == case.expected_primary_anchor, render_provider_mismatch(
        case,
        actual_verdict=response.verdict,
        actual_primary_anchor=response.primary_anchor,
        actual_roles=actual_roles,
        actual_anchors=actual_anchors,
    )
    assert actual_roles == case.expected_support_roles, render_provider_mismatch(
        case,
        actual_verdict=response.verdict,
        actual_primary_anchor=response.primary_anchor,
        actual_roles=actual_roles,
        actual_anchors=actual_anchors,
    )
    assert actual_anchors == expected_anchors, render_provider_mismatch(
        case,
        actual_verdict=response.verdict,
        actual_primary_anchor=response.primary_anchor,
        actual_roles=actual_roles,
        actual_anchors=actual_anchors,
    )


def _build_claim(text: str) -> ClaimUnit:
    return ClaimUnit(
        assertion_id=uuid.uuid4(),
        text=text,
        normalized_text=normalize_for_match(text),
        claim_type=ClaimType.FACT,
        sequence_order=1,
        support_status=SupportStatus.UNVERIFIED,
    )


def _build_segment(item: VerificationSupportItemGold) -> Segment:
    return Segment(
        id=uuid.uuid4(),
        source_document_id=uuid.uuid4(),
        page_start=item.page_start,
        line_start=item.line_start,
        page_end=item.page_end,
        line_end=item.line_end,
        raw_text=item.raw_text,
        normalized_text=normalize_for_match(item.raw_text),
        speaker=item.speaker,
        segment_type=item.segment_type,
    )


def _build_link(segment: Segment) -> SupportLink:
    return SupportLink(
        id=uuid.uuid4(),
        claim_unit_id=uuid.uuid4(),
        segment_id=segment.id,
        sequence_order=1,
        link_type=LinkType.MANUAL,
        citation_text=None,
        user_confirmed=True,
        segment=segment,
    )


def _build_provider_support_item(item: VerificationSupportItemGold) -> ProviderSupportItem:
    assert item.anchor is not None, "Provider gold cases require fully anchored support items."
    return ProviderSupportItem(
        segment_id=uuid.uuid4(),
        anchor=item.anchor,
        evidence_role=determine_evidence_role(
            segment_type=item.segment_type,
            speaker=item.speaker,
            raw_text=item.raw_text,
        ),
        speaker=item.speaker,
        segment_type=item.segment_type,
        raw_text=item.raw_text,
        normalized_text=normalize_for_match(item.raw_text),
    )


def _render_context_bundle(items: Sequence[VerificationSupportItemGold]) -> str:
    parts: list[str] = []
    for item in items:
        anchor = item.anchor or "unanchored"
        speaker = item.speaker or "UNKNOWN"
        parts.append(f"Anchor: {anchor}\nSpeaker: {speaker}\nText: {item.raw_text}")
    return "\n\n".join(parts)


def test_verification_gold_summary_is_clean() -> None:
    heuristic_report = run_verification_heuristic_gold_evaluation()
    provider_report = run_verification_provider_gold_evaluation()

    assert heuristic_report.total_cases == len(VERIFICATION_GOLD_CASES)
    assert not heuristic_report.failures, render_suite_report(heuristic_report)

    expected_provider_cases = len([case for case in VERIFICATION_GOLD_CASES if case.expected_provider_verdict is not None])
    assert provider_report.total_cases == expected_provider_cases
    assert not provider_report.failures, render_suite_report(provider_report)
