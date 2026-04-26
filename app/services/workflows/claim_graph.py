"""Deterministic cross-claim relationship inference for draft review runs."""

from collections.abc import Iterable
from dataclasses import dataclass
import re
from uuid import UUID

from app.core.enums import ClaimGraphRelationshipType, ClaimType
from app.models import ClaimUnit
from app.repositories.claim_graph_edges import ClaimGraphEdgeCreate
from app.services.parsing.normalization import normalize_for_match

TOKEN_RE = re.compile(r"[a-z0-9']+")
NEGATION_TOKENS = {"not", "no", "never", "none", "without", "cannot", "can't", "didn't", "didnt", "won't", "wont"}
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
    "was",
    "were",
    "is",
    "are",
    "do",
    "does",
    "did",
    "be",
    "been",
    "by",
    "with",
    "as",
    "at",
    "from",
}


@dataclass(slots=True)
class ClaimGraphSummary:
    edges: list[ClaimGraphEdgeCreate]


@dataclass(slots=True)
class _PreparedClaim:
    claim_id: UUID
    claim_type: ClaimType
    text: str
    normalized_text: str
    token_set: set[str]
    core_token_set: set[str]
    has_negation: bool


def build_claim_graph_edges(
    *,
    draft_id: UUID,
    draft_review_run_id: UUID,
    claims: Iterable[ClaimUnit],
) -> ClaimGraphSummary:
    prepared_claims = [_prepare_claim(claim) for claim in claims]
    edges: list[ClaimGraphEdgeCreate] = []

    for index, left in enumerate(prepared_claims):
        for right in prepared_claims[index + 1 :]:
            if not left.token_set or not right.token_set:
                continue

            if left.normalized_text == right.normalized_text:
                edges.extend(
                    _bidirectional_edges(
                        draft_id=draft_id,
                        draft_review_run_id=draft_review_run_id,
                        left=left,
                        right=right,
                        relationship_type=ClaimGraphRelationshipType.DUPLICATE_OF,
                        reason_code="exact_normalized_duplicate",
                        reason_text="Claims normalize to the same proposition.",
                        confidence_score=1.0,
                    )
                )
                continue

            if _claims_contradict(left, right):
                edges.extend(
                    _bidirectional_edges(
                        draft_id=draft_id,
                        draft_review_run_id=draft_review_run_id,
                        left=left,
                        right=right,
                        relationship_type=ClaimGraphRelationshipType.CONTRADICTS,
                        reason_code="negation_conflict",
                        reason_text="Claims share the same core proposition but opposite polarity.",
                        confidence_score=0.91,
                    )
                )
                continue

            if _depends_on(left, right):
                edges.append(
                    ClaimGraphEdgeCreate(
                        draft_id=draft_id,
                        draft_review_run_id=draft_review_run_id,
                        source_claim_id=left.claim_id,
                        target_claim_id=right.claim_id,
                        relationship_type=ClaimGraphRelationshipType.DEPENDS_ON,
                        reason_code="broader_claim_depends_on_narrower_claim",
                        reason_text="Broader claim appears to rely on a narrower embedded proposition.",
                        confidence_score=0.73,
                    )
                )
            elif _depends_on(right, left):
                edges.append(
                    ClaimGraphEdgeCreate(
                        draft_id=draft_id,
                        draft_review_run_id=draft_review_run_id,
                        source_claim_id=right.claim_id,
                        target_claim_id=left.claim_id,
                        relationship_type=ClaimGraphRelationshipType.DEPENDS_ON,
                        reason_code="broader_claim_depends_on_narrower_claim",
                        reason_text="Broader claim appears to rely on a narrower embedded proposition.",
                        confidence_score=0.73,
                    )
                )

            if _supports(left, right):
                edges.append(
                    ClaimGraphEdgeCreate(
                        draft_id=draft_id,
                        draft_review_run_id=draft_review_run_id,
                        source_claim_id=left.claim_id,
                        target_claim_id=right.claim_id,
                        relationship_type=ClaimGraphRelationshipType.SUPPORTS,
                        reason_code="fact_supports_inference",
                        reason_text="A factual claim materially overlaps an inference-level claim.",
                        confidence_score=0.68,
                    )
                )
            if _supports(right, left):
                edges.append(
                    ClaimGraphEdgeCreate(
                        draft_id=draft_id,
                        draft_review_run_id=draft_review_run_id,
                        source_claim_id=right.claim_id,
                        target_claim_id=left.claim_id,
                        relationship_type=ClaimGraphRelationshipType.SUPPORTS,
                        reason_code="fact_supports_inference",
                        reason_text="A factual claim materially overlaps an inference-level claim.",
                        confidence_score=0.68,
                    )
                )

    return ClaimGraphSummary(edges=_dedupe_edges(edges))


def _prepare_claim(claim: ClaimUnit) -> _PreparedClaim:
    normalized_text = normalize_for_match(claim.text)
    tokens = [token for token in TOKEN_RE.findall(normalized_text) if token not in STOPWORDS]
    token_set = set(tokens)
    has_negation = any(token in NEGATION_TOKENS for token in tokens)
    core_token_set = {token for token in token_set if token not in NEGATION_TOKENS}
    return _PreparedClaim(
        claim_id=claim.id,
        claim_type=claim.claim_type,
        text=claim.text,
        normalized_text=normalized_text,
        token_set=token_set,
        core_token_set=core_token_set,
        has_negation=has_negation,
    )


def _claims_contradict(left: _PreparedClaim, right: _PreparedClaim) -> bool:
    if left.has_negation == right.has_negation:
        return False
    if not left.core_token_set or not right.core_token_set:
        return False
    smaller = min(len(left.core_token_set), len(right.core_token_set))
    if smaller < 2:
        return False
    overlap = len(left.core_token_set & right.core_token_set) / smaller
    return overlap >= 0.8


def _depends_on(source: _PreparedClaim, target: _PreparedClaim) -> bool:
    if source.claim_id == target.claim_id:
        return False
    if len(source.core_token_set) < len(target.core_token_set) + 2:
        return False
    if not target.core_token_set:
        return False
    if not target.core_token_set.issubset(source.core_token_set):
        return False
    return True


def _supports(source: _PreparedClaim, target: _PreparedClaim) -> bool:
    if source.claim_type not in {ClaimType.FACT, ClaimType.QUOTE, ClaimType.MIXED}:
        return False
    if target.claim_type not in {ClaimType.INFERENCE, ClaimType.MIXED}:
        return False
    if not source.core_token_set or not target.core_token_set:
        return False
    overlap = len(source.core_token_set & target.core_token_set) / len(target.core_token_set)
    return overlap >= 0.6


def _bidirectional_edges(
    *,
    draft_id: UUID,
    draft_review_run_id: UUID,
    left: _PreparedClaim,
    right: _PreparedClaim,
    relationship_type: ClaimGraphRelationshipType,
    reason_code: str,
    reason_text: str,
    confidence_score: float,
) -> list[ClaimGraphEdgeCreate]:
    return [
        ClaimGraphEdgeCreate(
            draft_id=draft_id,
            draft_review_run_id=draft_review_run_id,
            source_claim_id=left.claim_id,
            target_claim_id=right.claim_id,
            relationship_type=relationship_type,
            reason_code=reason_code,
            reason_text=reason_text,
            confidence_score=confidence_score,
        ),
        ClaimGraphEdgeCreate(
            draft_id=draft_id,
            draft_review_run_id=draft_review_run_id,
            source_claim_id=right.claim_id,
            target_claim_id=left.claim_id,
            relationship_type=relationship_type,
            reason_code=reason_code,
            reason_text=reason_text,
            confidence_score=confidence_score,
        ),
    ]


def _dedupe_edges(edges: list[ClaimGraphEdgeCreate]) -> list[ClaimGraphEdgeCreate]:
    deduped: dict[tuple[str, str, str, str], ClaimGraphEdgeCreate] = {}
    for edge in edges:
        key = (
            str(edge.source_claim_id),
            str(edge.target_claim_id),
            edge.relationship_type.value,
            edge.reason_code or "",
        )
        deduped.setdefault(key, edge)
    return list(deduped.values())
