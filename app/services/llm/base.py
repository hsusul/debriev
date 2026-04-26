"""LLM provider abstractions for verification."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
from uuid import UUID

from app.core.enums import SupportStatus
from app.services.parsing.normalization import normalize_for_match
from app.services.verification.evidence_roles import (
    EvidenceRole,
    is_non_substantive_evidence_role,
    is_substantive_evidence_role,
    is_weak_substantive_text,
)


class ProviderUnavailableError(RuntimeError):
    """Raised when a configured provider cannot complete a verification call."""


TOKEN_RE = re.compile(r"[a-z0-9']+")
STOPWORDS = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "that", "this", "was", "were"}


@dataclass(slots=True)
class ProviderSupportItem:
    """Provider-facing support item derived from structured claim context."""

    segment_id: UUID
    anchor: str
    evidence_role: EvidenceRole
    speaker: str | None
    segment_type: str
    raw_text: str
    normalized_text: str


@dataclass(slots=True)
class ProviderRequest:
    """Provider input for claim verification."""

    claim_text: str
    support_items: list[ProviderSupportItem]
    context: str
    citations: list[str]
    heuristic_flags: list[str]


@dataclass(slots=True)
class ProviderSupportAssessment:
    """Provider assessment for a single support item."""

    segment_id: UUID
    anchor: str
    role: str
    contribution: str


@dataclass(slots=True)
class ProviderResponse:
    """Provider output for claim verification."""

    verdict: SupportStatus
    reasoning: str
    suggested_fix: str | None
    confidence_score: float | None
    support_assessments: list[ProviderSupportAssessment] = field(default_factory=list)
    primary_anchor: str | None = None


@dataclass(slots=True)
class _ScoredSupportItem:
    item: ProviderSupportItem
    raw_overlap: float
    overlap: float
    effective_score: float
    weak_substantive: bool


@dataclass(slots=True)
class _SupportScoringSummary:
    assessments: list[ProviderSupportAssessment]
    primary_anchor: str | None
    primary_overlap: float
    combined_overlap: float
    best_substantive_overlap: float
    best_contextual_overlap: float
    contextual_only: bool
    all_substantive_support_weak: bool
    support_tension: bool
    likely_narrow_support: bool
    contextual_dominant: bool


class VerificationProvider(ABC):
    """Abstract verification provider interface."""

    name: str
    model_version: str

    @abstractmethod
    def verify(self, request: ProviderRequest) -> ProviderResponse:
        """Classify a claim against linked context."""


def build_placeholder_provider_response(request: ProviderRequest, *, provider_label: str) -> ProviderResponse:
    """Build a deterministic provider response from ordered support items."""

    scoring = assess_support_items(
        request.claim_text,
        request.support_items,
    )
    verdict = _derive_placeholder_verdict(scoring)
    reasoning = _build_placeholder_reasoning(provider_label, scoring)
    suggested_fix = None
    if verdict in {SupportStatus.PARTIALLY_SUPPORTED, SupportStatus.AMBIGUOUS, SupportStatus.UNSUPPORTED}:
        suggested_fix = "Review the strongest support item and add better-linked testimony if needed."

    return ProviderResponse(
        verdict=verdict,
        reasoning=reasoning,
        suggested_fix=suggested_fix,
        confidence_score=_bucket_placeholder_confidence(verdict),
        support_assessments=scoring.assessments,
        primary_anchor=scoring.primary_anchor,
    )


def assess_support_items(
    claim_text: str,
    support_items: list[ProviderSupportItem],
) -> _SupportScoringSummary:
    """Assess ordered support items with simple lexical overlap and evidence-role weighting."""

    if not support_items:
        return _SupportScoringSummary(
            assessments=[],
            primary_anchor=None,
            primary_overlap=0.0,
            combined_overlap=0.0,
            best_substantive_overlap=0.0,
            best_contextual_overlap=0.0,
            contextual_only=False,
            all_substantive_support_weak=False,
            support_tension=False,
            likely_narrow_support=False,
            contextual_dominant=False,
        )

    claim_tokens = _tokenize(claim_text)
    scored_items: list[_ScoredSupportItem] = []

    for item in support_items:
        item_tokens = _tokenize(item.normalized_text or item.raw_text)
        raw_overlap = _lexical_overlap(claim_tokens, item_tokens)
        weak_substantive = _is_weak_substantive_item(item)
        overlap = _adjust_support_overlap(raw_overlap, weak_substantive=weak_substantive)
        effective_score = overlap + _support_priority_bonus(item, weak_substantive=weak_substantive)
        scored_items.append(
            _ScoredSupportItem(
                item=item,
                raw_overlap=raw_overlap,
                overlap=overlap,
                effective_score=effective_score,
                weak_substantive=weak_substantive,
            )
        )

    primary_index = max(
        range(len(scored_items)),
        key=lambda index: (scored_items[index].effective_score, scored_items[index].overlap, -index),
    )
    primary_item = scored_items[primary_index]
    primary_anchor = primary_item.item.anchor
    primary_overlap = primary_item.overlap
    combined_overlap = _lexical_overlap(
        claim_tokens,
        {token for item in support_items for token in _tokenize(item.normalized_text or item.raw_text)},
    )
    best_substantive_overlap = max(
        (scored.overlap for scored in scored_items if _is_substantive_item(scored.item)),
        default=0.0,
    )
    best_contextual_overlap = max(
        (scored.overlap for scored in scored_items if _is_non_substantive_item(scored.item)),
        default=0.0,
    )
    contextual_only = best_contextual_overlap > 0 and best_substantive_overlap < 0.2
    all_substantive_support_weak = _all_substantive_items_are_weak(scored_items)
    support_tension = _has_support_tension(scored_items)
    likely_narrow_support = _has_narrow_placeholder_support(
        primary_overlap=primary_overlap,
        combined_overlap=combined_overlap,
        support_count=len(support_items),
    )
    contextual_dominant = best_contextual_overlap >= 0.45 and best_substantive_overlap < 0.2

    assessments: list[ProviderSupportAssessment] = []
    for index, scored in enumerate(scored_items):
        role = _role_for_item(
            scored.item,
            is_primary=index == primary_index,
            overlap=scored.overlap,
            weak_substantive=scored.weak_substantive,
        )
        assessments.append(
            ProviderSupportAssessment(
                segment_id=scored.item.segment_id,
                anchor=scored.item.anchor,
                role=role,
                contribution=_contribution_text(
                    scored.item,
                    role,
                    scored.overlap,
                    likely_narrow_support=likely_narrow_support,
                    contextual_dominant=contextual_dominant,
                    weak_substantive=scored.weak_substantive,
                ),
            )
        )

    return _SupportScoringSummary(
        assessments=assessments,
        primary_anchor=primary_anchor,
        primary_overlap=primary_overlap,
        combined_overlap=combined_overlap,
        best_substantive_overlap=best_substantive_overlap,
        best_contextual_overlap=best_contextual_overlap,
        contextual_only=contextual_only,
        all_substantive_support_weak=all_substantive_support_weak,
        support_tension=support_tension,
        likely_narrow_support=likely_narrow_support,
        contextual_dominant=contextual_dominant,
    )


def _tokenize(text: str) -> set[str]:
    return {token for token in TOKEN_RE.findall(normalize_for_match(text)) if token not in STOPWORDS}


def _lexical_overlap(claim_tokens: set[str], support_tokens: set[str]) -> float:
    if not claim_tokens or not support_tokens:
        return 0.0
    return len(claim_tokens & support_tokens) / len(claim_tokens)


def _adjust_support_overlap(overlap: float, *, weak_substantive: bool) -> float:
    if weak_substantive:
        return min(overlap, 0.05)
    return overlap


def _support_priority_bonus(item: ProviderSupportItem, *, weak_substantive: bool) -> float:
    if weak_substantive:
        return -0.25
    if is_substantive_evidence_role(item.evidence_role):
        return 0.15
    if item.evidence_role == "contextual":
        return -0.15
    return 0.0


def _role_for_item(
    item: ProviderSupportItem,
    *,
    is_primary: bool,
    overlap: float,
    weak_substantive: bool,
) -> str:
    if weak_substantive:
        return "contextual"
    if is_non_substantive_evidence_role(item.evidence_role):
        return "contextual"
    if is_primary and overlap >= 0.25:
        return "primary"
    if is_substantive_evidence_role(item.evidence_role) and overlap >= 0.3:
        return "secondary"
    return "contextual"


def _contribution_text(
    item: ProviderSupportItem,
    role: str,
    overlap: float,
    *,
    likely_narrow_support: bool,
    contextual_dominant: bool,
    weak_substantive: bool,
) -> str:
    overlap_label = _overlap_label(overlap)
    source_label = _source_label(item)
    tail = ""
    if weak_substantive:
        tail = " The substantive statement is equivocal and does not state the fact directly."
    elif likely_narrow_support and role in {"primary", "secondary"}:
        tail = " Covers only a narrow slice of the claim."
    elif contextual_dominant and is_non_substantive_evidence_role(item.evidence_role):
        tail = " Mostly frames or contextualizes the issue rather than stating the fact directly."
    if role == "primary":
        return f"Primary support with {overlap_label} from {source_label}.{tail}"
    if role == "secondary":
        return f"Secondary support with {overlap_label} from {source_label}.{tail}"
    return f"Contextual support with {overlap_label} from {source_label}.{tail}"


def _overlap_label(overlap: float) -> str:
    if overlap >= 0.7:
        return "strong lexical overlap"
    if overlap >= 0.35:
        return "moderate lexical overlap"
    if overlap > 0:
        return "limited lexical overlap"
    return "no direct lexical overlap"


def _source_label(item: ProviderSupportItem) -> str:
    if item.evidence_role == "substantive":
        return "substantive record support"
    if item.evidence_role == "contextual":
        return "contextual record support"
    return "linked record text"


def _derive_placeholder_verdict(scoring: _SupportScoringSummary) -> SupportStatus:
    if scoring.contextual_only:
        return SupportStatus.AMBIGUOUS if scoring.combined_overlap > 0 else SupportStatus.UNSUPPORTED
    if scoring.all_substantive_support_weak:
        return SupportStatus.AMBIGUOUS if scoring.combined_overlap > 0 else SupportStatus.UNSUPPORTED
    if scoring.contextual_dominant and scoring.best_substantive_overlap < 0.15:
        return SupportStatus.AMBIGUOUS if scoring.combined_overlap > 0 else SupportStatus.UNSUPPORTED
    if scoring.support_tension and scoring.primary_overlap >= 0.45:
        return SupportStatus.PARTIALLY_SUPPORTED
    if (
        scoring.primary_overlap >= 0.7
        and scoring.best_substantive_overlap >= 0.55
        and not scoring.likely_narrow_support
    ):
        return SupportStatus.SUPPORTED
    if (
        scoring.best_substantive_overlap >= 0.35
        or scoring.primary_overlap >= 0.45
        or scoring.combined_overlap >= 0.55
    ):
        return SupportStatus.PARTIALLY_SUPPORTED
    if scoring.combined_overlap > 0:
        return SupportStatus.AMBIGUOUS
    return SupportStatus.UNSUPPORTED


def _build_placeholder_reasoning(
    provider_label: str,
    scoring: _SupportScoringSummary,
) -> str:
    parts = [f"{provider_label} placeholder ranked linked support items using overlap and evidence-role weighting."]
    if scoring.contextual_only:
        parts.append("Only contextual or non-substantive support materially overlaps the claim.")
    elif scoring.all_substantive_support_weak:
        parts.append("All substantive support is equivocal, so the support remains cautious.")
    elif scoring.support_tension:
        parts.append("Linked substantive support includes both direct and equivocal statements, so the result remains cautious.")
    elif scoring.contextual_dominant:
        parts.append("Contextual framing overlaps the claim more strongly than substantive support, so the result remains cautious.")
    elif scoring.likely_narrow_support:
        parts.append("Support is distributed across multiple linked items and appears narrower than the full claim.")
    if scoring.primary_anchor:
        parts.append(f"Primary support anchor: {scoring.primary_anchor}.")
    if scoring.assessments:
        rendered = "; ".join(
            f"{assessment.anchor} [{assessment.role}]: {assessment.contribution}"
            for assessment in scoring.assessments
        )
        parts.append(f"Support item assessments: {rendered}.")
    return " ".join(parts)


def _is_substantive_item(item: ProviderSupportItem) -> bool:
    return is_substantive_evidence_role(item.evidence_role)


def _is_non_substantive_item(item: ProviderSupportItem) -> bool:
    return is_non_substantive_evidence_role(item.evidence_role)


def _is_weak_substantive_item(item: ProviderSupportItem) -> bool:
    if not _is_substantive_item(item):
        return False
    return is_weak_substantive_text(item.raw_text)


def _has_narrow_placeholder_support(
    *,
    primary_overlap: float,
    combined_overlap: float,
    support_count: int,
) -> bool:
    if support_count < 2:
        return False
    return combined_overlap >= 0.55 and primary_overlap < 0.7 and (combined_overlap - primary_overlap) >= 0.2


def _all_substantive_items_are_weak(scored_items: list[_ScoredSupportItem]) -> bool:
    substantive_items = [scored for scored in scored_items if _is_substantive_item(scored.item)]
    if not substantive_items:
        return False
    return all(scored.weak_substantive for scored in substantive_items)


def _has_support_tension(scored_items: list[_ScoredSupportItem]) -> bool:
    strong_substantive_present = any(
        _is_substantive_item(scored.item) and not scored.weak_substantive and scored.overlap >= 0.55
        for scored in scored_items
    )
    overlapping_weak_substantive_present = any(
        _is_substantive_item(scored.item) and scored.weak_substantive and scored.raw_overlap >= 0.35
        for scored in scored_items
    )
    return strong_substantive_present and overlapping_weak_substantive_present


def _bucket_placeholder_confidence(verdict: SupportStatus) -> float:
    if verdict == SupportStatus.SUPPORTED:
        return 0.75
    if verdict == SupportStatus.PARTIALLY_SUPPORTED:
        return 0.6
    if verdict == SupportStatus.AMBIGUOUS:
        return 0.4
    return 0.25
