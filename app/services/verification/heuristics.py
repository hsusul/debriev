"""Deterministic verification heuristics."""

from dataclasses import dataclass
import re
from typing import Iterable

from app.core.enums import SupportStatus
from app.models import ClaimUnit, Segment, SupportLink
from app.services.parsing.normalization import normalize_for_match
from app.services.verification.evidence_roles import (
    determine_evidence_role,
    is_non_substantive_evidence_role,
    is_substantive_evidence_role,
    is_weak_substantive_text,
)

TOKEN_RE = re.compile(r"[a-z0-9']+")
QUOTE_RE = re.compile(r'"([^"\n]+)"|“([^”\n]+)”')
PROPER_TOKEN_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
ABSOLUTE_QUALIFIERS = {"always", "never", "all", "only", "entirely", "completely", "every", "none"}
KNOWLEDGE_MARKERS = ("knew", "should have known", "understood", "aware", "intended")
CAUSATION_MARKERS = ("caused", "because", "due to", "resulted in", "led to")
REPEATED_SCOPE_PATTERNS = (
    re.compile(r"\balways\b"),
    re.compile(r"\brepeatedly\b"),
    re.compile(r"\broutinely\b"),
    re.compile(r"\bongoing\b"),
    re.compile(r"\bcontinued?\b"),
    re.compile(r"\bcontinues?\b"),
    re.compile(r"\bregularly\b"),
    re.compile(r"\bdaily\b"),
    re.compile(r"\bweekly\b"),
    re.compile(r"\bmonthly\b"),
    re.compile(r"\bevery day\b"),
    re.compile(r"\beach day\b"),
)
DISCRETE_SCOPE_PATTERNS = (
    re.compile(r"\bonce\b"),
    re.compile(r"\bthat day\b"),
    re.compile(r"\bone time\b"),
    re.compile(r"\bsingle\b"),
    re.compile(r"\bon\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b"),
    re.compile(r"\bon\s+\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(r"\bon\s+\d{4}-\d{2}-\d{2}\b"),
)
STOPWORDS = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "that", "this", "was", "were"}
NAME_STOPWORDS = {
    "page",
    "question",
    "answer",
    "court",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "january",
    "february",
}
VERB_HINTS = {
    "is",
    "are",
    "was",
    "were",
    "signed",
    "approved",
    "reviewed",
    "caused",
    "opened",
    "closed",
    "sent",
    "received",
    "emailed",
    "observed",
    "knew",
    "understood",
    "failed",
    "continued",
    "admitted",
    "explained",
}


@dataclass(slots=True)
class HeuristicAssessment:
    """Structured deterministic verification output."""

    flags: list[str]
    verdict: SupportStatus
    reasoning: list[str]
    suggested_fix: str | None
    confidence_score: float | None


def evaluate_heuristics(
    claim: ClaimUnit,
    links: Iterable[SupportLink],
    segments: Iterable[Segment],
) -> HeuristicAssessment:
    """Run deterministic verification checks before any provider-backed classification."""

    linked_segments = list(segments)
    linked_links = list(links)
    flags: list[str] = []
    reasoning: list[str] = []
    suggested_fix: str | None = None

    if not linked_links:
        return HeuristicAssessment(
            flags=["missing_citation"],
            verdict=SupportStatus.UNVERIFIED,
            reasoning=["No support links are attached to this claim."],
            suggested_fix="Attach at least one anchored record segment before verification.",
            confidence_score=0.15,
        )

    invalid_anchor_count = sum(1 for segment in linked_segments if _has_invalid_anchor(segment))
    if invalid_anchor_count:
        flags.append("invalid_anchor")
        reasoning.append(f"{invalid_anchor_count} linked segment(s) are missing usable provenance anchors.")
        suggested_fix = "Review source parsing and relink the claim to segments with stable provenance."

    combined_text = " ".join(segment.normalized_text for segment in linked_segments)
    claim_text = normalize_for_match(claim.text)
    combined_raw_text = " ".join(segment.raw_text for segment in linked_segments)
    combined_overlap = _lexical_overlap(claim_text, combined_text)
    segment_overlaps = [(segment, _support_overlap_for_segment(claim_text, segment)) for segment in linked_segments]
    best_substantive_overlap = max(
        (overlap for segment, overlap in segment_overlaps if _is_substantive_support_segment(segment)),
        default=0.0,
    )
    best_contextual_overlap = max(
        (overlap for segment, overlap in segment_overlaps if _is_non_substantive_support_segment(segment)),
        default=0.0,
    )
    has_weak_substantive_support = any(
        _is_weak_substantive_segment(segment)
        for segment in linked_segments
        if _is_substantive_support_segment(segment)
    )
    all_substantive_support_weak = _all_substantive_support_is_weak(linked_segments)

    claim_qualifiers = {token for token in TOKEN_RE.findall(claim_text) if token in ABSOLUTE_QUALIFIERS}
    if claim_qualifiers and not claim_qualifiers.intersection(TOKEN_RE.findall(combined_text)):
        flags.append("absolute_qualifier_mismatch")
        reasoning.append("The claim uses absolute qualifiers not reflected in the linked record text.")
        suggested_fix = suggested_fix or "Soften the qualifier or add record support for the broader assertion."

    quoted_phrases = _extract_quoted_phrases(claim.text)
    if quoted_phrases and not all(_quoted_phrase_matches(phrase, combined_text) for phrase in quoted_phrases):
        flags.append("quote_mismatch_placeholder")
        reasoning.append("Quoted language does not appear verbatim in the linked text.")
        suggested_fix = suggested_fix or "Check the quote against the transcript and adjust the citation."

    if _has_subject_mismatch(claim.text, combined_raw_text):
        flags.append("subject_mismatch")
        reasoning.append("Named subject tokens in the claim do not match the linked record text.")
        suggested_fix = suggested_fix or "Relink the claim to testimony about the same actor or witness."

    if _has_temporal_scope_mismatch(claim_text, combined_text):
        flags.append("temporal_scope_mismatch")
        reasoning.append("The claim suggests repeated or ongoing conduct, but the support looks like a discrete event.")
        suggested_fix = suggested_fix or "Narrow the claim's time scope or add broader record support."

    knowledge_claim = any(marker in claim_text for marker in KNOWLEDGE_MARKERS)
    causation_claim = any(marker in claim_text for marker in CAUSATION_MARKERS)
    if knowledge_claim and not any(marker in combined_text for marker in KNOWLEDGE_MARKERS):
        flags.append("knowledge_escalation")
        reasoning.append("The claim attributes knowledge or intent beyond what the linked testimony states directly.")
        suggested_fix = suggested_fix or "Reframe the claim as an inference or add testimony addressing knowledge directly."
    if causation_claim and not any(marker in combined_text for marker in CAUSATION_MARKERS):
        flags.append("causation_escalation")
        reasoning.append("The claim states causation more directly than the linked testimony appears to support.")
        suggested_fix = suggested_fix or "Narrow the causal assertion or add testimony that ties the event to the outcome."
    if knowledge_claim or causation_claim:
        _append_unique(flags, "needs_human_review")
        reasoning.append("Knowledge or causation language should be reviewed by a human before acceptance.")

    if _has_contextual_only_support(
        linked_segments,
        best_substantive_overlap=best_substantive_overlap,
        best_contextual_overlap=best_contextual_overlap,
        has_weak_substantive_support=has_weak_substantive_support,
    ):
        flags.append("contextual_support_only")
        reasoning.append("Linked support is mostly contextual framing rather than substantive record support.")
        suggested_fix = suggested_fix or "Add anchored support that states the proposition directly."

    if _has_narrow_support(
        claim_text,
        combined_overlap=combined_overlap,
        best_substantive_overlap=best_substantive_overlap,
    ):
        flags.append("narrow_support")
        reasoning.append("The linked support appears narrower than the broader claim language.")
        suggested_fix = suggested_fix or "Split the claim or narrow it to the portion stated directly in the record."

    if not reasoning:
        reasoning.append("No deterministic red flags were triggered.")

    if all_substantive_support_weak:
        reasoning.append("Substantive support is equivocal across the linked segments and does not state the fact directly.")
        suggested_fix = suggested_fix or "Add direct record support that states the proposition without hedging."

    verdict = _derive_verdict(flags, combined_overlap, all_substantive_support_weak=all_substantive_support_weak)
    if verdict == SupportStatus.SUPPORTED:
        reasoning.append("Linked record text materially overlaps the claim language.")
    elif verdict == SupportStatus.PARTIALLY_SUPPORTED:
        reasoning.append("Linked record text appears related but narrower than the claim.")
    elif verdict == SupportStatus.OVERSTATED:
        reasoning.append("Linked record text supports part of the claim, but the drafted proposition goes further.")
    elif verdict == SupportStatus.UNSUPPORTED:
        reasoning.append("Linked record text has minimal lexical overlap with the claim.")

    return HeuristicAssessment(
        flags=flags,
        verdict=verdict,
        reasoning=reasoning,
        suggested_fix=suggested_fix,
        confidence_score=_bucket_confidence(verdict, flags, combined_overlap),
    )


def _has_invalid_anchor(segment: Segment) -> bool:
    return not segment.has_usable_anchor


def _lexical_overlap(claim_text: str, combined_text: str) -> float:
    claim_tokens = {token for token in TOKEN_RE.findall(claim_text) if token not in STOPWORDS}
    combined_tokens = {token for token in TOKEN_RE.findall(combined_text) if token not in STOPWORDS}
    if not claim_tokens or not combined_tokens:
        return 0.0
    return len(claim_tokens & combined_tokens) / len(claim_tokens)


def _support_overlap_for_segment(claim_text: str, segment: Segment) -> float:
    overlap = _lexical_overlap(claim_text, segment.normalized_text)
    if _is_weak_substantive_segment(segment):
        return min(overlap, 0.05)
    return overlap


def _derive_verdict(flags: list[str], overlap: float, *, all_substantive_support_weak: bool = False) -> SupportStatus:
    if "missing_citation" in flags:
        return SupportStatus.UNVERIFIED
    if "invalid_anchor" in flags:
        return SupportStatus.AMBIGUOUS
    if "quote_mismatch_placeholder" in flags:
        return SupportStatus.OVERSTATED
    if "contextual_support_only" in flags:
        return SupportStatus.AMBIGUOUS if overlap > 0 else SupportStatus.UNSUPPORTED
    if all_substantive_support_weak:
        return SupportStatus.AMBIGUOUS if overlap > 0 else SupportStatus.UNSUPPORTED
    if "knowledge_escalation" in flags or "causation_escalation" in flags:
        return SupportStatus.AMBIGUOUS if overlap >= 0.15 else SupportStatus.UNSUPPORTED
    if "subject_mismatch" in flags:
        return SupportStatus.AMBIGUOUS if overlap >= 0.25 else SupportStatus.UNSUPPORTED
    if "narrow_support" in flags:
        return SupportStatus.OVERSTATED if overlap >= 0.25 else SupportStatus.AMBIGUOUS
    if "absolute_qualifier_mismatch" in flags or "temporal_scope_mismatch" in flags:
        return SupportStatus.OVERSTATED if overlap >= 0.45 else SupportStatus.AMBIGUOUS
    if overlap >= 0.7 and "needs_human_review" not in flags:
        return SupportStatus.SUPPORTED
    if overlap >= 0.4:
        return SupportStatus.PARTIALLY_SUPPORTED if "needs_human_review" not in flags else SupportStatus.AMBIGUOUS
    if overlap >= 0.15:
        return SupportStatus.AMBIGUOUS
    return SupportStatus.UNSUPPORTED


def _extract_quoted_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    for straight_quote, curly_quote in QUOTE_RE.findall(text):
        phrase = straight_quote or curly_quote
        cleaned = normalize_for_match(phrase)
        if cleaned:
            phrases.append(cleaned)
    return phrases


def _quoted_phrase_matches(phrase: str, combined_text: str) -> bool:
    if phrase in combined_text:
        return True

    phrase_tokens = TOKEN_RE.findall(phrase)
    combined_tokens = TOKEN_RE.findall(combined_text)
    if not phrase_tokens:
        return True

    window_size = len(phrase_tokens)
    for index in range(len(combined_tokens) - window_size + 1):
        if combined_tokens[index : index + window_size] == phrase_tokens:
            return True
    return False


def _has_subject_mismatch(claim_text: str, combined_raw_text: str) -> bool:
    claim_tokens = _extract_named_tokens(claim_text)
    combined_tokens = _extract_named_tokens(combined_raw_text)
    if not claim_tokens or not combined_tokens:
        return False
    return claim_tokens.isdisjoint(combined_tokens)


def _extract_named_tokens(text: str) -> set[str]:
    tokens = {
        token.lower()
        for token in PROPER_TOKEN_RE.findall(text)
        if token.lower() not in NAME_STOPWORDS and token not in {"Q", "A"}
    }
    return tokens


def _has_temporal_scope_mismatch(claim_text: str, combined_text: str) -> bool:
    claim_repeated = any(pattern.search(claim_text) for pattern in REPEATED_SCOPE_PATTERNS)
    combined_repeated = any(pattern.search(combined_text) for pattern in REPEATED_SCOPE_PATTERNS)
    combined_discrete = any(pattern.search(combined_text) for pattern in DISCRETE_SCOPE_PATTERNS)
    return claim_repeated and not combined_repeated and combined_discrete


def _has_contextual_only_support(
    segments: list[Segment],
    *,
    best_substantive_overlap: float,
    best_contextual_overlap: float,
    has_weak_substantive_support: bool,
) -> bool:
    if not segments:
        return False
    has_substantive_segment = any(_is_substantive_support_segment(segment) for segment in segments)
    if not has_substantive_segment and best_contextual_overlap > 0:
        return True
    if has_weak_substantive_support and best_contextual_overlap >= 0.35 and best_substantive_overlap < 0.2:
        return True
    return best_contextual_overlap >= 0.4 and best_substantive_overlap < 0.2


def _has_narrow_support(
    claim_text: str,
    *,
    combined_overlap: float,
    best_substantive_overlap: float,
) -> bool:
    if combined_overlap < 0.25:
        return False
    if best_substantive_overlap < 0.2:
        return False
    if best_substantive_overlap >= 0.7:
        return False
    if not _has_multiple_predicates(claim_text):
        return False
    if best_substantive_overlap <= 0.6:
        return True
    return best_substantive_overlap < combined_overlap


def _all_substantive_support_is_weak(segments: list[Segment]) -> bool:
    substantive_segments = [segment for segment in segments if _is_substantive_support_segment(segment)]
    if not substantive_segments:
        return False
    return all(_is_weak_substantive_segment(segment) for segment in substantive_segments)


def _has_multiple_predicates(text: str) -> bool:
    normalized = normalize_for_match(text)
    tokens = TOKEN_RE.findall(normalized)
    if "and" not in tokens:
        return False
    verb_count = sum(1 for token in tokens if token in VERB_HINTS or token.endswith("ed"))
    return verb_count >= 2


def _segment_evidence_role(segment: Segment) -> str:
    return determine_evidence_role(
        segment_type=segment.segment_type,
        speaker=segment.speaker,
        raw_text=segment.raw_text,
    )


def _is_substantive_support_segment(segment: Segment) -> bool:
    return is_substantive_evidence_role(_segment_evidence_role(segment))


def _is_non_substantive_support_segment(segment: Segment) -> bool:
    return is_non_substantive_evidence_role(_segment_evidence_role(segment))


def _is_weak_substantive_segment(segment: Segment) -> bool:
    if not _is_substantive_support_segment(segment):
        return False
    return is_weak_substantive_text(segment.raw_text)


def _append_unique(flags: list[str], flag: str) -> None:
    if flag not in flags:
        flags.append(flag)


def _bucket_confidence(verdict: SupportStatus, flags: list[str], overlap: float) -> float:
    if "missing_citation" in flags:
        return 0.15
    if "invalid_anchor" in flags:
        return 0.3
    if "contextual_support_only" in flags:
        return 0.35
    if verdict == SupportStatus.SUPPORTED:
        return 0.85
    if verdict == SupportStatus.PARTIALLY_SUPPORTED:
        return 0.65
    if verdict == SupportStatus.OVERSTATED:
        return 0.55 if overlap >= 0.45 else 0.4
    if verdict == SupportStatus.AMBIGUOUS:
        return 0.4
    if verdict in {SupportStatus.UNSUPPORTED, SupportStatus.CONTRADICTED}:
        return 0.25
    return 0.35
