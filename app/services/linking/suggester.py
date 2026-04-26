"""Simple lexical support-link suggestion helpers."""

from dataclasses import dataclass
import re
from typing import Iterable
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.exceptions import NotFoundError
from app.repositories.claims import ClaimsRepository
from app.repositories.segments import SegmentRepository
from app.models import Segment
from app.services.parsing.normalization import normalize_for_match

WORD_RE = re.compile(r"[a-z0-9']+")
STOPWORDS = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "that", "this"}


@dataclass(slots=True)
class SuggestedLink:
    """Ranked segment suggestion."""

    segment_id: object
    score: float
    rationale: str


class SupportSuggestionService:
    """Draft-scoped support suggestion entry points."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def suggest_for_draft(
        self,
        draft_id: UUID,
        claim_text: str,
        *,
        limit: int = 5,
        fallback_to_matter_sources: bool = True,
    ) -> list[SuggestedLink]:
        segments = SegmentRepository(self.session).list_candidate_segments_for_draft(
            draft_id,
            fallback_to_matter_sources=fallback_to_matter_sources,
        )
        return suggest_segments(claim_text, segments, limit=limit)

    def suggest_for_claim(
        self,
        claim_id: UUID,
        *,
        limit: int = 5,
        fallback_to_matter_sources: bool = True,
    ) -> list[SuggestedLink]:
        claim = ClaimsRepository(self.session).get(claim_id)
        if claim is None:
            raise NotFoundError("Claim unit not found.")

        segments = SegmentRepository(self.session).list_candidate_segments_for_claim(
            claim_id,
            fallback_to_matter_sources=fallback_to_matter_sources,
        )
        return suggest_segments(claim.text, segments, limit=limit)


def suggest_segments(claim_text: str, segments: Iterable[Segment], *, limit: int = 5) -> list[SuggestedLink]:
    """Rank segments by simple lexical overlap."""

    claim_tokens = {token for token in WORD_RE.findall(normalize_for_match(claim_text)) if token not in STOPWORDS}
    suggestions: list[SuggestedLink] = []

    for segment in segments:
        segment_tokens = {token for token in WORD_RE.findall(segment.normalized_text) if token not in STOPWORDS}
        if not claim_tokens or not segment_tokens:
            continue
        overlap = len(claim_tokens & segment_tokens) / len(claim_tokens)
        if overlap <= 0:
            continue
        suggestions.append(
            SuggestedLink(
                segment_id=segment.id,
                score=round(overlap, 2),
                rationale="Lexical overlap between claim terms and normalized segment text.",
            )
        )

    return sorted(suggestions, key=lambda item: item.score, reverse=True)[:limit]
