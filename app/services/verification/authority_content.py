"""Narrow MVP authority-content retrieval and deterministic support checking."""

from dataclasses import dataclass
import re

from app.core.enums import StructuredReasoningCategory, SupportStatus
from app.services.parsing.case_citations import AuthorityMatch
from app.services.parsing.normalization import normalize_for_match, normalize_text

TOKEN_RE = re.compile(r"[a-z0-9']+")
LEADING_CITATION_SIGNAL_RE = re.compile(r"^(?:Under|See|But see|Cf\.)\s+", re.IGNORECASE)
LEADING_PROPOSITION_VERB_RE = re.compile(
    r"^(?:held(?:\s+that)?|holds(?:\s+that)?|establishes(?:\s+that)?|provides(?:\s+that)?|requires(?:\s+that)?)\s+",
    re.IGNORECASE,
)
ABSOLUTE_QUALIFIERS = {"always", "never", "all", "only", "must", "every", "none"}
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "that",
    "this",
    "those",
    "these",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "its",
    "as",
    "by",
    "with",
    "from",
    "at",
    "into",
    "their",
    "them",
    "they",
    "he",
    "she",
    "his",
    "her",
    "public",
    "schools",
    "school",
    "education",
    "court",
    "case",
}


@dataclass(slots=True, frozen=True)
class AuthorityContentEntry:
    authority_id: str
    canonical_name: str
    canonical_citation: str
    source_name: str
    excerpts: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class AuthorityContentEvaluation:
    authority_content_status: str
    proposition_verdict: SupportStatus | None
    reasoning: str | None
    reasoning_categories: list[str]
    confidence_score: float | None
    authority_excerpt: str | None
    support_verification_basis: str | None
    suggested_fix: str | None


_AUTHORITY_CONTENT_CATALOG = {
    "brown-v-board-of-education-347-us-483": AuthorityContentEntry(
        authority_id="brown-v-board-of-education-347-us-483",
        canonical_name="Brown v. Board of Education",
        canonical_citation="Brown v. Board of Education, 347 U.S. 483 (1954)",
        source_name="debriev_mvp_authority_content_catalog",
        excerpts=(
            "Separate educational facilities are inherently unequal.",
            "Segregation of children in public schools solely on the basis of race deprives minority children of equal educational opportunities.",
        ),
    ),
    "celotex-v-catrett-477-us-317": AuthorityContentEntry(
        authority_id="celotex-v-catrett-477-us-317",
        canonical_name="Celotex Corp. v. Catrett",
        canonical_citation="Celotex Corp. v. Catrett, 477 U.S. 317 (1986)",
        source_name="debriev_mvp_authority_content_catalog",
        excerpts=(
            "The plain language of Rule 56(c) mandates the entry of summary judgment against a party who fails to make a showing sufficient to establish the existence of an element essential to that party's case.",
            "The burden on the moving party may be discharged by showing that there is an absence of evidence to support the nonmoving party's case.",
        ),
    ),
}


def get_authority_content(matched_authority: AuthorityMatch) -> AuthorityContentEntry | None:
    """Return repo-local MVP authority content for a matched authority when available."""

    return _AUTHORITY_CONTENT_CATALOG.get(matched_authority.authority_id)


def evaluate_proposition_against_authority_content(
    proposition_text: str,
    citation_text: str,
    matched_authority: AuthorityMatch | None,
) -> AuthorityContentEvaluation:
    """Check proposition support against matched authority content when available."""

    if matched_authority is None:
        return AuthorityContentEvaluation(
            authority_content_status="not_applicable",
            proposition_verdict=None,
            reasoning=None,
            reasoning_categories=[],
            confidence_score=None,
            authority_excerpt=None,
            support_verification_basis=None,
            suggested_fix=None,
        )

    content = get_authority_content(matched_authority)
    if content is None:
        return AuthorityContentEvaluation(
            authority_content_status="unavailable",
            proposition_verdict=None,
            reasoning=(
                "Authority matched, but no authority content is available in the MVP catalog for proposition verification."
            ),
            reasoning_categories=[StructuredReasoningCategory.WEAK_SUPPORT.value],
            confidence_score=0.55,
            authority_excerpt=None,
            support_verification_basis="authority_content_unavailable",
            suggested_fix="Add authority text coverage for this case or verify the proposition manually.",
        )

    proposition_core = _extract_proposition_core(proposition_text, citation_text)
    if not proposition_core:
        return AuthorityContentEvaluation(
            authority_content_status="available",
            proposition_verdict=SupportStatus.AMBIGUOUS,
            reasoning="Authority content is available, but the proposition text is too thin to verify deterministically.",
            reasoning_categories=[StructuredReasoningCategory.WEAK_SUPPORT.value],
            confidence_score=0.4,
            authority_excerpt=content.excerpts[0],
            support_verification_basis="authority_content_insufficient_proposition_text",
            suggested_fix="State the proposition more explicitly so it can be compared against the authority content.",
        )

    best_excerpt, best_overlap = _best_excerpt_match(proposition_core, content.excerpts)
    proposition_tokens = _content_tokens(proposition_core)
    excerpt_tokens = _content_tokens(best_excerpt) if best_excerpt else set()
    qualifier_mismatch = _has_qualifier_mismatch(proposition_tokens, excerpt_tokens)

    if best_excerpt and best_overlap >= 0.55 and not qualifier_mismatch:
        return AuthorityContentEvaluation(
            authority_content_status="support_verified",
            proposition_verdict=SupportStatus.SUPPORTED,
            reasoning="Matched authority content materially overlaps the drafted proposition.",
            reasoning_categories=[],
            confidence_score=0.86,
            authority_excerpt=best_excerpt,
            support_verification_basis="authority_content_deterministic_supported",
            suggested_fix=None,
        )

    if best_excerpt and best_overlap >= 0.28:
        verdict = SupportStatus.OVERSTATED if qualifier_mismatch else SupportStatus.AMBIGUOUS
        reasoning = (
            "Authority content is available, but the proposition goes further than the matched excerpt."
            if qualifier_mismatch
            else "Authority content is available, but the matched excerpt only partially overlaps the proposition."
        )
        suggested_fix = (
            "Narrow the proposition to match the authority excerpt more closely."
            if qualifier_mismatch
            else "Quote or narrow the proposition to the portion stated more directly in the authority excerpt."
        )
        return AuthorityContentEvaluation(
            authority_content_status="available",
            proposition_verdict=verdict,
            reasoning=reasoning,
            reasoning_categories=[StructuredReasoningCategory.WEAK_SUPPORT.value],
            confidence_score=0.64 if qualifier_mismatch else 0.58,
            authority_excerpt=best_excerpt,
            support_verification_basis="authority_content_deterministic_partial_overlap",
            suggested_fix=suggested_fix,
        )

    return AuthorityContentEvaluation(
        authority_content_status="available",
        proposition_verdict=SupportStatus.UNSUPPORTED,
        reasoning="Authority content is available, but no excerpt clearly supports the drafted proposition.",
        reasoning_categories=[StructuredReasoningCategory.WEAK_SUPPORT.value],
        confidence_score=0.74,
        authority_excerpt=best_excerpt or content.excerpts[0],
        support_verification_basis="authority_content_deterministic_no_clear_support",
        suggested_fix="Revise the proposition to track the authority more closely or verify it manually.",
    )


def _extract_proposition_core(proposition_text: str, citation_text: str) -> str:
    normalized = normalize_text(proposition_text)
    normalized = LEADING_CITATION_SIGNAL_RE.sub("", normalized)

    if normalized.startswith(citation_text):
        normalized = normalized[len(citation_text) :].lstrip(" ,")
    normalized = LEADING_PROPOSITION_VERB_RE.sub("", normalized)
    return normalized.strip(" .;:")


def _best_excerpt_match(proposition_text: str, excerpts: tuple[str, ...]) -> tuple[str | None, float]:
    best_excerpt: str | None = None
    best_overlap = 0.0
    proposition_tokens = _content_tokens(proposition_text)
    if not proposition_tokens:
        return None, 0.0

    for excerpt in excerpts:
        overlap = _lexical_overlap(proposition_tokens, _content_tokens(excerpt))
        if overlap > best_overlap:
            best_excerpt = excerpt
            best_overlap = overlap
    return best_excerpt, best_overlap


def _content_tokens(text: str) -> set[str]:
    normalized = normalize_for_match(text)
    return {
        token
        for token in TOKEN_RE.findall(normalized)
        if token not in STOPWORDS and len(token) > 2
    }


def _lexical_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)


def _has_qualifier_mismatch(proposition_tokens: set[str], excerpt_tokens: set[str]) -> bool:
    proposition_qualifiers = proposition_tokens & ABSOLUTE_QUALIFIERS
    if not proposition_qualifiers:
        return False
    return not proposition_qualifiers.issubset(excerpt_tokens)
