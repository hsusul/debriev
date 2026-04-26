from uuid import uuid4

from app.core.enums import ClaimGraphRelationshipType, ClaimType, SupportStatus
from app.models import ClaimUnit
from app.services.parsing.normalization import normalize_for_match
from app.services.workflows.claim_graph import build_claim_graph_edges


def build_claim(text: str, *, claim_type: ClaimType = ClaimType.FACT) -> ClaimUnit:
    return ClaimUnit(
        id=uuid4(),
        assertion_id=uuid4(),
        text=text,
        normalized_text=normalize_for_match(text),
        claim_type=claim_type,
        sequence_order=1,
        support_status=SupportStatus.UNVERIFIED,
    )


def test_claim_graph_builder_detects_contradiction_dependency_and_support_edges() -> None:
    claim_one = build_claim("Doe delivered the notice.")
    claim_two = build_claim("Doe never delivered the notice.")
    claim_three = build_claim("Doe delivered the notice and approved the invoice.")
    claim_four = build_claim("Doe likely delivered the notice.", claim_type=ClaimType.INFERENCE)

    result = build_claim_graph_edges(
        draft_id=uuid4(),
        draft_review_run_id=uuid4(),
        claims=[claim_one, claim_two, claim_three, claim_four],
    )

    relationships = {
        (edge.source_claim_id, edge.target_claim_id, edge.relationship_type)
        for edge in result.edges
    }

    assert (
        claim_one.id,
        claim_two.id,
        ClaimGraphRelationshipType.CONTRADICTS,
    ) in relationships
    assert (
        claim_two.id,
        claim_one.id,
        ClaimGraphRelationshipType.CONTRADICTS,
    ) in relationships
    assert (
        claim_three.id,
        claim_one.id,
        ClaimGraphRelationshipType.DEPENDS_ON,
    ) in relationships
    assert (
        claim_one.id,
        claim_four.id,
        ClaimGraphRelationshipType.SUPPORTS,
    ) in relationships
