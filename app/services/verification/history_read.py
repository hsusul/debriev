"""Shared read-side builders for verification history surfaces."""

from app.schemas.verification import VerificationRunRead
from app.services.verification.snapshot_adapter import parse_verification_support_snapshot
from app.services.verification.snapshot_mapper import map_parsed_snapshot_to_history_fields


def build_verification_run_history_read(run) -> VerificationRunRead:
    parsed_snapshot = parse_verification_support_snapshot(run.support_snapshot, run.support_snapshot_version)
    history_fields = map_parsed_snapshot_to_history_fields(parsed_snapshot)
    return VerificationRunRead(
        id=run.id,
        claim_unit_id=run.claim_unit_id,
        model_version=run.model_version,
        prompt_version=run.prompt_version,
        deterministic_flags=list(run.deterministic_flags),
        reasoning_categories=list(run.reasoning_categories),
        verdict=run.verdict,
        reasoning=run.reasoning,
        suggested_fix=run.suggested_fix,
        confidence_score=run.confidence_score,
        created_at=run.created_at,
        support_snapshot_status=history_fields.support_snapshot_status,
        support_snapshot_note=history_fields.support_snapshot_note,
        support_snapshot_version=history_fields.support_snapshot_version,
        support_snapshot=history_fields.support_snapshot,
    )
