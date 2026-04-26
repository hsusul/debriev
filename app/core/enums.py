"""Domain enumerations."""

from enum import Enum


class SourceType(str, Enum):
    DEPOSITION = "DEPOSITION"
    EXHIBIT = "EXHIBIT"
    DECLARATION = "DECLARATION"


class ParserStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class DraftMode(str, Enum):
    DRAFT = "DRAFT"
    COMPILE = "COMPILE"
    AUDIT = "AUDIT"


class ClaimType(str, Enum):
    FACT = "FACT"
    INFERENCE = "INFERENCE"
    QUOTE = "QUOTE"
    MIXED = "MIXED"


class SupportStatus(str, Enum):
    UNVERIFIED = "UNVERIFIED"
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    OVERSTATED = "OVERSTATED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    AMBIGUOUS = "AMBIGUOUS"


class LinkType(str, Enum):
    MANUAL = "MANUAL"
    AUTO_SUGGESTED = "AUTO_SUGGESTED"
    AUTO_ACCEPTED = "AUTO_ACCEPTED"


class ClaimReviewAction(str, Enum):
    ACKNOWLEDGE_RISK = "acknowledge_risk"
    MARK_FOR_REVISION = "mark_for_revision"
    RESOLVE_WITH_EDIT = "resolve_with_edit"


class DecisionAction(str, Enum):
    ACKNOWLEDGE_INFERENCE = "ACKNOWLEDGE_INFERENCE"
    INTENTIONAL_ADVOCACY = "INTENTIONAL_ADVOCACY"
    NEEDS_CITATION_LATER = "NEEDS_CITATION_LATER"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    ESCALATE_FOR_REVIEW = "ESCALATE_FOR_REVIEW"


class ReExtractionRunKind(str, Enum):
    PREVIEW = "PREVIEW"
    APPLY = "APPLY"


class DraftReviewRunStatus(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class StructuredReasoningCategory(str, Enum):
    TEMPORAL_MISMATCH = "temporal_mismatch"
    SCOPE_MISMATCH = "scope_mismatch"
    WEAK_SUPPORT = "weak_support"
    CONTRADICTION = "contradiction"
    MISSING_AUTHORITY = "missing_authority"
    FABRICATED_AUTHORITY = "fabricated_authority"


class ClaimGraphRelationshipType(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    DEPENDS_ON = "depends_on"
    DUPLICATE_OF = "duplicate_of"
