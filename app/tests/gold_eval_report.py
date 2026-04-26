from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import uuid
from typing import Literal

from app.core.enums import ClaimType, LinkType, SupportStatus
from app.models.claim_unit import ClaimUnit
from app.models.segment import Segment
from app.models.support_link import SupportLink
from app.services.claims.extractor import extract_claim_candidates
from app.services.llm.base import ProviderRequest, ProviderSupportItem
from app.services.llm.openai_provider import OpenAIProvider
from app.services.parsing.normalization import normalize_for_match
from app.services.verification.evidence_roles import determine_evidence_role
from app.services.verification.heuristics import evaluate_heuristics
from app.tests.gold_sets import (
    EXTRACTION_GOLD_CASES,
    VERIFICATION_GOLD_CASES,
    ExtractionGoldCase,
    VerificationGoldCase,
    VerificationSupportItemGold,
)


@dataclass(frozen=True, slots=True)
class GoldCaseFailure:
    case_name: str
    case_category: str
    mismatch_kinds: tuple[str, ...]
    details: str


@dataclass(frozen=True, slots=True)
class GoldSuiteReport:
    suite_name: str
    total_cases: int
    case_category_counts: tuple[tuple[str, int], ...]
    failures: tuple[GoldCaseFailure, ...]

    @property
    def passed_cases(self) -> int:
        return self.total_cases - len(self.failures)

    @property
    def failed_cases(self) -> int:
        return len(self.failures)


@dataclass(frozen=True, slots=True)
class GoldEvaluationSummary:
    extraction: GoldSuiteReport
    verification_heuristics: GoldSuiteReport
    verification_provider: GoldSuiteReport

    @property
    def total_cases(self) -> int:
        return (
            self.extraction.total_cases
            + self.verification_heuristics.total_cases
            + self.verification_provider.total_cases
        )

    @property
    def passed_cases(self) -> int:
        return (
            self.extraction.passed_cases
            + self.verification_heuristics.passed_cases
            + self.verification_provider.passed_cases
        )

    @property
    def failed_cases(self) -> int:
        return self.total_cases - self.passed_cases


@dataclass(frozen=True, slots=True)
class ExtractionCaseResult:
    case: ExtractionGoldCase
    actual_texts: tuple[str, ...]
    actual_types: tuple[ClaimType, ...]
    mismatch_kinds: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.mismatch_kinds

    @property
    def details(self) -> str:
        return render_extraction_mismatch(
            self.case,
            actual_texts=self.actual_texts,
            actual_types=self.actual_types,
        )


@dataclass(frozen=True, slots=True)
class VerificationHeuristicCaseResult:
    case: VerificationGoldCase
    actual_flags: tuple[str, ...]
    actual_verdict: SupportStatus
    mismatch_kinds: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.mismatch_kinds

    @property
    def details(self) -> str:
        return render_heuristic_mismatch(
            self.case,
            actual_flags=self.actual_flags,
            actual_verdict=self.actual_verdict,
        )


@dataclass(frozen=True, slots=True)
class VerificationProviderCaseResult:
    case: VerificationGoldCase
    actual_verdict: SupportStatus
    actual_primary_anchor: str | None
    actual_roles: tuple[str, ...]
    actual_anchors: tuple[str, ...]
    mismatch_kinds: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.mismatch_kinds

    @property
    def details(self) -> str:
        return render_provider_mismatch(
            self.case,
            actual_verdict=self.actual_verdict,
            actual_primary_anchor=self.actual_primary_anchor,
            actual_roles=self.actual_roles,
            actual_anchors=self.actual_anchors,
        )


def evaluate_extraction_case(
    case: ExtractionGoldCase,
    *,
    mode: Literal["auto", "legacy", "structured"] = "auto",
) -> ExtractionCaseResult:
    claims = extract_claim_candidates(case.assertion_text, mode=mode)
    actual_texts = tuple(claim.text for claim in claims)
    actual_types = tuple(claim.claim_type for claim in claims)
    mismatch_kinds: list[str] = []

    if actual_texts != case.expected_claim_texts:
        mismatch_kinds.append("extraction_text_mismatch")
    if actual_types != case.expected_claim_types:
        mismatch_kinds.append("claim_type_mismatch")

    return ExtractionCaseResult(
        case=case,
        actual_texts=actual_texts,
        actual_types=actual_types,
        mismatch_kinds=tuple(mismatch_kinds),
    )


def evaluate_verification_heuristic_case(case: VerificationGoldCase) -> VerificationHeuristicCaseResult:
    claim = _build_claim(case.claim_text)
    segments = [_build_segment(item) for item in case.support_items]
    links = [_build_link(segment) for segment in segments]

    result = evaluate_heuristics(claim, links=links, segments=segments)
    actual_flags = tuple(sorted(result.flags))
    expected_flags = tuple(sorted(case.expected_flags))
    mismatch_kinds: list[str] = []

    missing_flags = set(expected_flags) - set(actual_flags)
    unexpected_flags = set(actual_flags) - set(expected_flags)
    if missing_flags:
        mismatch_kinds.append("missing_heuristic_flags")
    if unexpected_flags:
        mismatch_kinds.append("unexpected_heuristic_flags")
    if result.verdict != case.expected_verdict:
        mismatch_kinds.append("verdict_mismatch")

    return VerificationHeuristicCaseResult(
        case=case,
        actual_flags=actual_flags,
        actual_verdict=result.verdict,
        mismatch_kinds=tuple(mismatch_kinds),
    )


def evaluate_verification_provider_case(case: VerificationGoldCase) -> VerificationProviderCaseResult:
    if case.expected_provider_verdict is None:
        raise ValueError(f"Provider evaluation is not defined for gold case '{case.name}'.")

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
    mismatch_kinds: list[str] = []

    if response.verdict != case.expected_provider_verdict:
        mismatch_kinds.append("verdict_mismatch")
    if response.primary_anchor != case.expected_primary_anchor:
        mismatch_kinds.append("primary_anchor_mismatch")
    if actual_roles != (case.expected_support_roles or ()):
        mismatch_kinds.append("support_role_mismatch")
    if actual_anchors != expected_anchors:
        mismatch_kinds.append("support_anchor_order_mismatch")

    return VerificationProviderCaseResult(
        case=case,
        actual_verdict=response.verdict,
        actual_primary_anchor=response.primary_anchor,
        actual_roles=actual_roles,
        actual_anchors=actual_anchors,
        mismatch_kinds=tuple(mismatch_kinds),
    )


def run_extraction_gold_evaluation(
    cases: Sequence[ExtractionGoldCase] = EXTRACTION_GOLD_CASES,
    *,
    mode: Literal["auto", "legacy", "structured"] = "auto",
) -> GoldSuiteReport:
    failures = tuple(
        _build_case_failure(result.case, result.mismatch_kinds, result.details)
        for result in (evaluate_extraction_case(case, mode=mode) for case in cases)
        if not result.passed
    )
    return _build_suite_report(f"Extraction gold set ({mode})", cases, failures)


def run_verification_heuristic_gold_evaluation(
    cases: Sequence[VerificationGoldCase] = VERIFICATION_GOLD_CASES,
) -> GoldSuiteReport:
    failures = tuple(
        _build_case_failure(result.case, result.mismatch_kinds, result.details)
        for result in (evaluate_verification_heuristic_case(case) for case in cases)
        if not result.passed
    )
    return _build_suite_report("Verification heuristic gold set", cases, failures)


def run_verification_provider_gold_evaluation(
    cases: Sequence[VerificationGoldCase] = VERIFICATION_GOLD_CASES,
) -> GoldSuiteReport:
    provider_cases = tuple(case for case in cases if case.expected_provider_verdict is not None)
    failures = tuple(
        _build_case_failure(result.case, result.mismatch_kinds, result.details)
        for result in (evaluate_verification_provider_case(case) for case in provider_cases)
        if not result.passed
    )
    return _build_suite_report("Verification provider gold set", provider_cases, failures)


def run_gold_evaluation() -> GoldEvaluationSummary:
    return GoldEvaluationSummary(
        extraction=run_extraction_gold_evaluation(),
        verification_heuristics=run_verification_heuristic_gold_evaluation(),
        verification_provider=run_verification_provider_gold_evaluation(),
    )


def render_gold_evaluation(summary: GoldEvaluationSummary) -> str:
    lines = [
        "Debriev gold-set evaluation",
        f"Overall: {summary.passed_cases}/{summary.total_cases} passed, {summary.failed_cases} failed",
        "",
        render_suite_report(summary.extraction),
        "",
        render_suite_report(summary.verification_heuristics),
        "",
        render_suite_report(summary.verification_provider),
    ]
    return "\n".join(lines)


def render_suite_report(report: GoldSuiteReport) -> str:
    lines = [
        f"{report.suite_name}: {report.passed_cases}/{report.total_cases} passed, {report.failed_cases} failed",
        f"Coverage: {_render_category_counts(report.case_category_counts)}",
    ]

    if not report.failures:
        lines.append("Failures: none")
        return "\n".join(lines)

    lines.append("Failure groups:")
    for mismatch_kind, failures in _group_failures_by_mismatch(report.failures).items():
        lines.append(f"  {mismatch_kind} ({len(failures)})")
        for failure in failures:
            lines.append(f"    {failure.case_name} [{failure.case_category}]")
            for detail_line in failure.details.splitlines():
                lines.append(f"      {detail_line}")

    return "\n".join(lines)


def render_extraction_mismatch(
    case: ExtractionGoldCase,
    *,
    actual_texts: Sequence[str],
    actual_types: Sequence[ClaimType],
) -> str:
    return (
        f"Case: {case.name} [{case.category}]\n"
        f"Input: {case.assertion_text}\n"
        f"Expected texts: {list(case.expected_claim_texts)}\n"
        f"Actual texts: {list(actual_texts)}\n"
        f"Expected types: {[claim_type.value for claim_type in case.expected_claim_types]}\n"
        f"Actual types: {[claim_type.value for claim_type in actual_types]}"
    )


def render_heuristic_mismatch(
    case: VerificationGoldCase,
    *,
    actual_flags: Sequence[str],
    actual_verdict: SupportStatus,
) -> str:
    expected_flags = set(case.expected_flags)
    actual_flag_set = set(actual_flags)
    missing_flags = sorted(expected_flags - actual_flag_set)
    unexpected_flags = sorted(actual_flag_set - expected_flags)
    return (
        f"Case: {case.name} [{case.category}]\n"
        f"Claim: {case.claim_text}\n"
        f"Support: {[item.raw_text for item in case.support_items]}\n"
        f"Expected flags: {sorted(case.expected_flags)}\n"
        f"Actual flags: {list(actual_flags)}\n"
        f"Missing flags: {missing_flags}\n"
        f"Unexpected flags: {unexpected_flags}\n"
        f"Expected verdict: {case.expected_verdict.value}\n"
        f"Actual verdict: {actual_verdict.value}"
    )


def render_provider_mismatch(
    case: VerificationGoldCase,
    *,
    actual_verdict: SupportStatus,
    actual_primary_anchor: str | None,
    actual_roles: Sequence[str],
    actual_anchors: Sequence[str],
) -> str:
    expected_anchors = [item.anchor for item in case.support_items if item.anchor is not None]
    return (
        f"Case: {case.name} [{case.category}]\n"
        f"Claim: {case.claim_text}\n"
        f"Support anchors: {expected_anchors}\n"
        f"Expected provider verdict: {case.expected_provider_verdict.value if case.expected_provider_verdict else None}\n"
        f"Actual provider verdict: {actual_verdict.value}\n"
        f"Expected primary anchor: {case.expected_primary_anchor}\n"
        f"Actual primary anchor: {actual_primary_anchor}\n"
        f"Expected roles: {list(case.expected_support_roles) if case.expected_support_roles else []}\n"
        f"Actual roles: {list(actual_roles)}\n"
        f"Expected support anchor order: {expected_anchors}\n"
        f"Actual support anchor order: {list(actual_anchors)}"
    )


def _build_suite_report(
    suite_name: str,
    cases: Sequence[ExtractionGoldCase | VerificationGoldCase],
    failures: tuple[GoldCaseFailure, ...],
) -> GoldSuiteReport:
    category_counts = Counter(case.category for case in cases)
    ordered_counts = tuple(sorted(category_counts.items(), key=lambda item: (-item[1], item[0])))
    return GoldSuiteReport(
        suite_name=suite_name,
        total_cases=len(cases),
        case_category_counts=ordered_counts,
        failures=failures,
    )


def _build_case_failure(
    case: ExtractionGoldCase | VerificationGoldCase,
    mismatch_kinds: Sequence[str],
    details: str,
) -> GoldCaseFailure:
    return GoldCaseFailure(
        case_name=case.name,
        case_category=case.category,
        mismatch_kinds=tuple(mismatch_kinds),
        details=details,
    )


def _group_failures_by_mismatch(failures: Sequence[GoldCaseFailure]) -> dict[str, list[GoldCaseFailure]]:
    grouped: dict[str, list[GoldCaseFailure]] = defaultdict(list)
    for failure in failures:
        for mismatch_kind in failure.mismatch_kinds:
            grouped[mismatch_kind].append(failure)
    return {key: grouped[key] for key in sorted(grouped)}


def _render_category_counts(counts: Iterable[tuple[str, int]]) -> str:
    return ", ".join(f"{category}={count}" for category, count in counts) or "none"


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


def main() -> int:
    report = run_gold_evaluation()
    print(render_gold_evaluation(report))
    return 1 if report.failed_cases else 0


if __name__ == "__main__":
    raise SystemExit(main())
