"""Structured reasoning-category classification for verification results."""

from app.core.enums import StructuredReasoningCategory, SupportStatus

FABRICATED_AUTHORITY_FLAGS = frozenset({"quote_mismatch_placeholder"})
MISSING_AUTHORITY_FLAGS = frozenset({"missing_citation"})
TEMPORAL_FLAGS = frozenset({"temporal_scope_mismatch"})
SCOPE_FLAGS = frozenset({"subject_mismatch", "absolute_qualifier_mismatch", "out_of_scope_support_link", "cross_matter_support_link"})
WEAK_SUPPORT_FLAGS = frozenset(
    {
        "contextual_support_only",
        "narrow_support",
        "knowledge_escalation",
        "causation_escalation",
        "needs_human_review",
        "invalid_anchor",
    }
)


def classify_reasoning_categories(
    *,
    deterministic_flags: list[str],
    verdict: SupportStatus,
) -> list[str]:
    """Map deterministic verification signals into stable reasoning categories."""

    categories: list[StructuredReasoningCategory] = []
    flags = set(deterministic_flags)

    if flags & FABRICATED_AUTHORITY_FLAGS:
        categories.append(StructuredReasoningCategory.FABRICATED_AUTHORITY)
    if flags & MISSING_AUTHORITY_FLAGS:
        categories.append(StructuredReasoningCategory.MISSING_AUTHORITY)
    if flags & TEMPORAL_FLAGS:
        categories.append(StructuredReasoningCategory.TEMPORAL_MISMATCH)
    if flags & SCOPE_FLAGS:
        categories.append(StructuredReasoningCategory.SCOPE_MISMATCH)
    if flags & WEAK_SUPPORT_FLAGS:
        categories.append(StructuredReasoningCategory.WEAK_SUPPORT)
    if verdict == SupportStatus.CONTRADICTED or any("contradict" in flag for flag in flags):
        categories.append(StructuredReasoningCategory.CONTRADICTION)

    return [category.value for category in categories]
