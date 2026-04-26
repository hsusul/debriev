"""Context assembly for verification providers."""

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable
from uuid import UUID

from app.models import ClaimUnit, Segment, SupportLink
from app.services.verification.evidence_roles import EvidenceRole, determine_evidence_role


@dataclass(slots=True)
class SupportItem:
    """Structured support item used to build verification context."""

    segment_id: UUID
    anchor: str
    evidence_role: EvidenceRole
    speaker: str | None
    segment_type: str
    raw_text: str
    normalized_text: str


@dataclass(slots=True)
class ClaimContext:
    """Structured claim context passed into verification providers."""

    claim_text: str
    support_items: list[SupportItem]
    segment_bundle: str
    citations: list[str]


def build_claim_context(claim: ClaimUnit, links: Iterable[SupportLink], segments: Iterable[Segment]) -> ClaimContext:
    """Assemble a deterministic support bundle for a claim."""

    segment_by_id = {segment.id: segment for segment in segments}
    support_items: list[SupportItem] = []

    for link in _ordered_links(links):
        segment = getattr(link, "segment", None) or segment_by_id.get(link.segment_id)
        if segment is None:
            continue

        anchor = link.citation_text or segment.rendered_anchor
        support_items.append(
            SupportItem(
                segment_id=segment.id,
                anchor=anchor,
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
        )

    citations = [item.anchor for item in support_items]
    segment_bundle = "\n\n".join(_format_support_item(index, item) for index, item in enumerate(support_items, start=1))
    return ClaimContext(
        claim_text=claim.text,
        support_items=support_items,
        segment_bundle=segment_bundle,
        citations=citations,
    )


def _ordered_links(links: Iterable[SupportLink]) -> list[SupportLink]:
    return sorted(
        links,
        key=lambda link: (
            link.sequence_order if link.sequence_order is not None else float("inf"),
            link.created_at or datetime.min,
            str(link.id),
        ),
    )


def _format_support_item(index: int, item: SupportItem) -> str:
    speaker = item.speaker or "UNKNOWN"
    return (
        f"[{index}] Anchor: {item.anchor} | Speaker: {speaker} | Type: {item.segment_type}\n"
        f"{item.raw_text}"
    )
