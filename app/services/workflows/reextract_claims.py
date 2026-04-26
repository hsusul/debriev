"""Explicit compare/apply workflow for assertion claim re-extraction."""

from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.enums import ClaimType
from app.core.exceptions import ConflictError, NotFoundError
from app.core.enums import ReExtractionRunKind
from app.models import CURRENT_REEXTRACTION_RUN_SNAPSHOT_VERSION, ClaimUnit
from app.repositories.assertions import AssertionRepository
from app.repositories.claims import ClaimsRepository
from app.repositories.drafts import DraftRepository
from app.repositories.reextraction_runs import ReExtractionRunRepository
from app.services.claims.extractor import (
    CURRENT_EXTRACTION_VERSION,
    DEFAULT_EXTRACTION_MODE,
    ExtractionMode,
    ExtractionStrategy,
    build_extraction_snapshot,
    extract_claim_candidates_with_strategy,
)

ExistingExtractionStatus = Literal["legacy_unversioned", "versioned"]
BatchPreviewStatus = Literal["ready", "unchanged", "blocked"]
BatchApplyStatus = Literal["applied", "skipped", "blocked"]


@dataclass(slots=True)
class ReExtractionClaimPreview:
    claim_id: UUID | None
    text: str
    normalized_text: str
    claim_type: ClaimType
    sequence_order: int


@dataclass(slots=True)
class ExistingExtractionMetadata:
    status: ExistingExtractionStatus
    strategy: str | None
    version: int | None
    snapshot_present: bool


@dataclass(slots=True)
class ProposedExtractionMetadata:
    strategy: ExtractionStrategy
    version: int


@dataclass(slots=True)
class AssertionReExtractionComparison:
    assertion_id: UUID
    existing_metadata: ExistingExtractionMetadata
    proposed_metadata: ProposedExtractionMetadata
    existing_claims: list[ReExtractionClaimPreview] = field(default_factory=list)
    proposed_claims: list[ReExtractionClaimPreview] = field(default_factory=list)
    materially_changed: bool = False
    apply_requires_replacement: bool = False
    can_apply: bool = True
    blocked_reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AssertionReExtractionApplyResult:
    assertion_id: UUID
    comparison: AssertionReExtractionComparison
    claims: list[ClaimUnit]
    metadata_updated: bool
    claims_replaced: bool


@dataclass(slots=True)
class DraftReExtractionPreviewItem:
    assertion_id: UUID
    paragraph_index: int | None
    sentence_index: int | None
    assertion_text: str
    status: BatchPreviewStatus
    existing_metadata: ExistingExtractionMetadata
    proposed_metadata: ProposedExtractionMetadata
    materially_changed: bool
    apply_requires_replacement: bool
    can_apply: bool
    blocked_reasons: list[str] = field(default_factory=list)
    existing_claim_count: int = 0
    proposed_claim_count: int = 0


@dataclass(slots=True)
class DraftReExtractionPreviewResult:
    run_id: UUID | None
    draft_id: UUID
    requested_mode: ExtractionMode
    total_assertions: int
    ready_assertions: int
    unchanged_assertions: int
    blocked_assertions: int
    materially_changed_assertions: int
    legacy_unversioned_assertions: int
    items: list[DraftReExtractionPreviewItem] = field(default_factory=list)


@dataclass(slots=True)
class DraftReExtractionApplyItem:
    assertion_id: UUID
    paragraph_index: int | None
    sentence_index: int | None
    assertion_text: str
    status: BatchApplyStatus
    existing_metadata: ExistingExtractionMetadata
    proposed_metadata: ProposedExtractionMetadata
    materially_changed: bool
    apply_requires_replacement: bool
    can_apply: bool
    claims_replaced: bool
    metadata_updated: bool
    blocked_reasons: list[str] = field(default_factory=list)
    resulting_claims: list[ReExtractionClaimPreview] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DraftReExtractionApplyResult:
    run_id: UUID | None
    draft_id: UUID
    requested_mode: ExtractionMode
    total_assertions: int
    applied_assertions: int
    skipped_assertions: int
    blocked_assertions: int
    replaced_assertions: int
    metadata_only_assertions: int
    items: list[DraftReExtractionApplyItem] = field(default_factory=list)


class ClaimReExtractionService:
    """Explicit compare/apply workflow for migrating persisted claim units."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.assertions = AssertionRepository(session)
        self.claims = ClaimsRepository(session)
        self.drafts = DraftRepository(session)
        self.reextraction_runs = ReExtractionRunRepository(session)

    def compare_assertion(
        self,
        assertion_id: UUID,
        *,
        mode: ExtractionMode = DEFAULT_EXTRACTION_MODE,
    ) -> AssertionReExtractionComparison:
        assertion = self.assertions.get(assertion_id)
        if assertion is None:
            raise NotFoundError("Assertion not found.")

        existing_claims = self.claims.list_by_assertion(assertion_id)
        proposed_plan = extract_claim_candidates_with_strategy(assertion.raw_text, mode=mode)

        existing_metadata = ExistingExtractionMetadata(
            status=_determine_existing_extraction_status(assertion),
            strategy=assertion.extraction_strategy,
            version=assertion.extraction_version,
            snapshot_present=assertion.extraction_snapshot is not None,
        )
        proposed_metadata = ProposedExtractionMetadata(
            strategy=proposed_plan.strategy,
            version=CURRENT_EXTRACTION_VERSION,
        )
        existing_previews = [_build_claim_preview(claim) for claim in existing_claims]
        proposed_previews = [_build_claim_preview(claim) for claim in proposed_plan.claims]
        materially_changed = _claim_preview_signature(existing_previews) != _claim_preview_signature(proposed_previews)
        apply_requires_replacement = materially_changed
        blocked_reasons = _build_apply_blockers(existing_claims) if apply_requires_replacement else []

        return AssertionReExtractionComparison(
            assertion_id=assertion.id,
            existing_metadata=existing_metadata,
            proposed_metadata=proposed_metadata,
            existing_claims=existing_previews,
            proposed_claims=proposed_previews,
            materially_changed=materially_changed,
            apply_requires_replacement=apply_requires_replacement,
            can_apply=not blocked_reasons,
            blocked_reasons=blocked_reasons,
        )

    def apply_assertion(
        self,
        assertion_id: UUID,
        *,
        mode: ExtractionMode = DEFAULT_EXTRACTION_MODE,
    ) -> AssertionReExtractionApplyResult:
        comparison = self.compare_assertion(assertion_id, mode=mode)
        assertion = self.assertions.get(assertion_id)
        if assertion is None:
            raise NotFoundError("Assertion not found.")

        if comparison.apply_requires_replacement and comparison.blocked_reasons:
            raise ConflictError(" ".join(comparison.blocked_reasons))

        persisted_claims = self.claims.list_by_assertion(assertion_id)
        claims_replaced = False

        if comparison.apply_requires_replacement:
            proposed_plan = extract_claim_candidates_with_strategy(assertion.raw_text, mode=mode)
            persisted_claims = self.claims.replace_for_assertion(assertion_id, proposed_plan.claims)
            claims_replaced = True

        snapshot = build_extraction_snapshot(
            assertion_id=assertion.id,
            source_assertion_text=assertion.raw_text,
            normalized_assertion_text=assertion.normalized_text,
            extraction_strategy=comparison.proposed_metadata.strategy,
            extraction_version=comparison.proposed_metadata.version,
            claims=persisted_claims,
        )
        self.assertions.record_extraction(
            assertion,
            strategy=comparison.proposed_metadata.strategy,
            version=comparison.proposed_metadata.version,
            snapshot=snapshot,
        )
        self.session.flush()

        return AssertionReExtractionApplyResult(
            assertion_id=assertion.id,
            comparison=comparison,
            claims=persisted_claims,
            metadata_updated=True,
            claims_replaced=claims_replaced,
        )

    def preview_draft(
        self,
        draft_id: UUID,
        *,
        mode: ExtractionMode = DEFAULT_EXTRACTION_MODE,
    ) -> DraftReExtractionPreviewResult:
        if self.drafts.get(draft_id) is None:
            raise NotFoundError("Draft not found.")

        assertions = self.assertions.list_by_draft(draft_id)
        items: list[DraftReExtractionPreviewItem] = []
        snapshot_items: list[dict[str, object]] = []
        ready_assertions = 0
        unchanged_assertions = 0
        blocked_assertions = 0
        materially_changed_assertions = 0
        legacy_unversioned_assertions = 0

        for assertion in assertions:
            comparison = self.compare_assertion(assertion.id, mode=mode)
            status = _preview_status_for_comparison(comparison)
            if status == "ready":
                ready_assertions += 1
            elif status == "unchanged":
                unchanged_assertions += 1
            else:
                blocked_assertions += 1

            if comparison.materially_changed:
                materially_changed_assertions += 1
            if comparison.existing_metadata.status == "legacy_unversioned":
                legacy_unversioned_assertions += 1

            items.append(
                DraftReExtractionPreviewItem(
                    assertion_id=assertion.id,
                    paragraph_index=assertion.paragraph_index,
                    sentence_index=assertion.sentence_index,
                    assertion_text=assertion.raw_text,
                    status=status,
                    existing_metadata=comparison.existing_metadata,
                    proposed_metadata=comparison.proposed_metadata,
                    materially_changed=comparison.materially_changed,
                    apply_requires_replacement=comparison.apply_requires_replacement,
                    can_apply=comparison.can_apply,
                    blocked_reasons=list(comparison.blocked_reasons),
                    existing_claim_count=len(comparison.existing_claims),
                    proposed_claim_count=len(comparison.proposed_claims),
                )
            )
            snapshot_items.append(
                _build_preview_snapshot_item(assertion, comparison)
            )

        result = DraftReExtractionPreviewResult(
            run_id=None,
            draft_id=draft_id,
            requested_mode=mode,
            total_assertions=len(assertions),
            ready_assertions=ready_assertions,
            unchanged_assertions=unchanged_assertions,
            blocked_assertions=blocked_assertions,
            materially_changed_assertions=materially_changed_assertions,
            legacy_unversioned_assertions=legacy_unversioned_assertions,
            items=items,
        )
        run = self.reextraction_runs.create(
            draft_id=draft_id,
            run_kind=ReExtractionRunKind.PREVIEW,
            requested_mode=mode,
            extraction_version=CURRENT_EXTRACTION_VERSION,
            total_assertions=result.total_assertions,
            ready_assertions=result.ready_assertions,
            unchanged_assertions=result.unchanged_assertions,
            applied_assertions=0,
            skipped_assertions=0,
            blocked_assertions=result.blocked_assertions,
            materially_changed_assertions=result.materially_changed_assertions,
            legacy_unversioned_assertions=result.legacy_unversioned_assertions,
            replaced_assertions=0,
            metadata_only_assertions=0,
            snapshot_version=CURRENT_REEXTRACTION_RUN_SNAPSHOT_VERSION,
            snapshot=_build_draft_run_snapshot(
                draft_id=draft_id,
                run_kind=ReExtractionRunKind.PREVIEW,
                requested_mode=mode,
                items=snapshot_items,
                summary={
                    "total_assertions": result.total_assertions,
                    "ready_assertions": result.ready_assertions,
                    "unchanged_assertions": result.unchanged_assertions,
                    "blocked_assertions": result.blocked_assertions,
                    "materially_changed_assertions": result.materially_changed_assertions,
                    "legacy_unversioned_assertions": result.legacy_unversioned_assertions,
                    "applied_assertions": 0,
                    "skipped_assertions": 0,
                    "replaced_assertions": 0,
                    "metadata_only_assertions": 0,
                },
            ),
        )
        result.run_id = run.id
        return result

    def apply_draft(
        self,
        draft_id: UUID,
        *,
        mode: ExtractionMode = DEFAULT_EXTRACTION_MODE,
    ) -> DraftReExtractionApplyResult:
        if self.drafts.get(draft_id) is None:
            raise NotFoundError("Draft not found.")

        assertions = self.assertions.list_by_draft(draft_id)
        items: list[DraftReExtractionApplyItem] = []
        snapshot_items: list[dict[str, object]] = []
        applied_assertions = 0
        skipped_assertions = 0
        blocked_assertions = 0
        replaced_assertions = 0
        metadata_only_assertions = 0

        for assertion in assertions:
            comparison = self.compare_assertion(assertion.id, mode=mode)
            if not _needs_apply(comparison):
                skipped_assertions += 1
                items.append(
                    DraftReExtractionApplyItem(
                        assertion_id=assertion.id,
                        paragraph_index=assertion.paragraph_index,
                        sentence_index=assertion.sentence_index,
                        assertion_text=assertion.raw_text,
                        status="skipped",
                        existing_metadata=comparison.existing_metadata,
                        proposed_metadata=comparison.proposed_metadata,
                        materially_changed=comparison.materially_changed,
                        apply_requires_replacement=comparison.apply_requires_replacement,
                        can_apply=comparison.can_apply,
                        claims_replaced=False,
                        metadata_updated=False,
                        resulting_claims=[_build_claim_preview(claim) for claim in self.claims.list_by_assertion(assertion.id)],
                        notes=["Assertion already matches the selected extraction strategy/version and needs no migration."],
                    )
                )
                snapshot_items.append(
                    _build_apply_snapshot_item(
                        assertion=assertion,
                        comparison=comparison,
                        status="skipped",
                        claims_replaced=False,
                        metadata_updated=False,
                        resulting_claims=self.claims.list_by_assertion(assertion.id),
                        notes=["Assertion already matches the selected extraction strategy/version and needs no migration."],
                    )
                )
                continue

            if not comparison.can_apply:
                blocked_assertions += 1
                items.append(
                    DraftReExtractionApplyItem(
                        assertion_id=assertion.id,
                        paragraph_index=assertion.paragraph_index,
                        sentence_index=assertion.sentence_index,
                        assertion_text=assertion.raw_text,
                        status="blocked",
                        existing_metadata=comparison.existing_metadata,
                        proposed_metadata=comparison.proposed_metadata,
                        materially_changed=comparison.materially_changed,
                        apply_requires_replacement=comparison.apply_requires_replacement,
                        can_apply=comparison.can_apply,
                        claims_replaced=False,
                        metadata_updated=False,
                        blocked_reasons=list(comparison.blocked_reasons),
                        resulting_claims=[_build_claim_preview(claim) for claim in self.claims.list_by_assertion(assertion.id)],
                        notes=["Assertion re-extraction was not applied because replacement is currently blocked."],
                    )
                )
                snapshot_items.append(
                    _build_apply_snapshot_item(
                        assertion=assertion,
                        comparison=comparison,
                        status="blocked",
                        claims_replaced=False,
                        metadata_updated=False,
                        resulting_claims=self.claims.list_by_assertion(assertion.id),
                        notes=["Assertion re-extraction was not applied because replacement is currently blocked."],
                    )
                )
                continue

            applied = self.apply_assertion(assertion.id, mode=mode)
            refreshed_assertion = self.assertions.get(assertion.id)
            if refreshed_assertion is None:
                raise NotFoundError("Assertion not found.")

            applied_assertions += 1
            if applied.claims_replaced:
                replaced_assertions += 1
            else:
                metadata_only_assertions += 1

            notes = [
                "Extraction metadata and snapshot were refreshed.",
            ]
            if applied.claims_replaced:
                notes.append("Persisted claim units were replaced during batch re-extraction.")
            else:
                notes.append("Existing claim units were retained during batch re-extraction.")

            items.append(
                DraftReExtractionApplyItem(
                    assertion_id=assertion.id,
                    paragraph_index=refreshed_assertion.paragraph_index,
                    sentence_index=refreshed_assertion.sentence_index,
                    assertion_text=refreshed_assertion.raw_text,
                    status="applied",
                    existing_metadata=comparison.existing_metadata,
                    proposed_metadata=comparison.proposed_metadata,
                    materially_changed=comparison.materially_changed,
                    apply_requires_replacement=comparison.apply_requires_replacement,
                    can_apply=comparison.can_apply,
                    claims_replaced=applied.claims_replaced,
                    metadata_updated=applied.metadata_updated,
                    resulting_claims=[_build_claim_preview(claim) for claim in applied.claims],
                    notes=notes,
                )
            )
            snapshot_items.append(
                _build_apply_snapshot_item(
                    assertion=refreshed_assertion,
                    comparison=comparison,
                    status="applied",
                    claims_replaced=applied.claims_replaced,
                    metadata_updated=applied.metadata_updated,
                    resulting_claims=applied.claims,
                    notes=notes,
                )
            )

        result = DraftReExtractionApplyResult(
            run_id=None,
            draft_id=draft_id,
            requested_mode=mode,
            total_assertions=len(assertions),
            applied_assertions=applied_assertions,
            skipped_assertions=skipped_assertions,
            blocked_assertions=blocked_assertions,
            replaced_assertions=replaced_assertions,
            metadata_only_assertions=metadata_only_assertions,
            items=items,
        )
        run = self.reextraction_runs.create(
            draft_id=draft_id,
            run_kind=ReExtractionRunKind.APPLY,
            requested_mode=mode,
            extraction_version=CURRENT_EXTRACTION_VERSION,
            total_assertions=result.total_assertions,
            ready_assertions=0,
            unchanged_assertions=0,
            applied_assertions=result.applied_assertions,
            skipped_assertions=result.skipped_assertions,
            blocked_assertions=result.blocked_assertions,
            materially_changed_assertions=sum(1 for item in result.items if item.materially_changed),
            legacy_unversioned_assertions=sum(
                1 for item in result.items if item.existing_metadata.status == "legacy_unversioned"
            ),
            replaced_assertions=result.replaced_assertions,
            metadata_only_assertions=result.metadata_only_assertions,
            snapshot_version=CURRENT_REEXTRACTION_RUN_SNAPSHOT_VERSION,
            snapshot=_build_draft_run_snapshot(
                draft_id=draft_id,
                run_kind=ReExtractionRunKind.APPLY,
                requested_mode=mode,
                items=snapshot_items,
                summary={
                    "total_assertions": result.total_assertions,
                    "ready_assertions": 0,
                    "unchanged_assertions": 0,
                    "blocked_assertions": result.blocked_assertions,
                    "materially_changed_assertions": sum(1 for item in result.items if item.materially_changed),
                    "legacy_unversioned_assertions": sum(
                        1 for item in result.items if item.existing_metadata.status == "legacy_unversioned"
                    ),
                    "applied_assertions": result.applied_assertions,
                    "skipped_assertions": result.skipped_assertions,
                    "replaced_assertions": result.replaced_assertions,
                    "metadata_only_assertions": result.metadata_only_assertions,
                },
            ),
        )
        result.run_id = run.id
        return result


def _build_claim_preview(claim: object) -> ReExtractionClaimPreview:
    return ReExtractionClaimPreview(
        claim_id=getattr(claim, "id", None),
        text=claim.text,
        normalized_text=claim.normalized_text,
        claim_type=claim.claim_type,
        sequence_order=claim.sequence_order,
    )


def _claim_preview_signature(claims: list[ReExtractionClaimPreview]) -> tuple[tuple[str, str, ClaimType, int], ...]:
    return tuple(
        (claim.text, claim.normalized_text, claim.claim_type, claim.sequence_order)
        for claim in claims
    )


def _determine_existing_extraction_status(assertion: object) -> ExistingExtractionStatus:
    if (
        getattr(assertion, "extraction_strategy", None) is None
        and getattr(assertion, "extraction_version", None) is None
        and getattr(assertion, "extraction_snapshot", None) is None
    ):
        return "legacy_unversioned"
    return "versioned"


def _build_apply_blockers(existing_claims: list[ClaimUnit]) -> list[str]:
    blocked_reasons: list[str] = []
    if any(claim.support_links for claim in existing_claims):
        blocked_reasons.append(
            "Re-extraction cannot replace persisted claim units while support links are attached."
        )
    if any(claim.verification_runs for claim in existing_claims):
        blocked_reasons.append(
            "Re-extraction cannot replace persisted claim units while verification runs are attached."
        )
    return blocked_reasons


def _needs_apply(comparison: AssertionReExtractionComparison) -> bool:
    return bool(
        comparison.materially_changed
        or comparison.existing_metadata.status == "legacy_unversioned"
        or comparison.existing_metadata.strategy != comparison.proposed_metadata.strategy
        or comparison.existing_metadata.version != comparison.proposed_metadata.version
        or not comparison.existing_metadata.snapshot_present
    )


def _preview_status_for_comparison(comparison: AssertionReExtractionComparison) -> BatchPreviewStatus:
    if _needs_apply(comparison) and not comparison.can_apply:
        return "blocked"
    if _needs_apply(comparison):
        return "ready"
    return "unchanged"


def _build_metadata_snapshot(metadata: ExistingExtractionMetadata | ProposedExtractionMetadata) -> dict[str, object]:
    payload: dict[str, object] = {
        "strategy": metadata.strategy,
        "version": metadata.version,
    }
    if isinstance(metadata, ExistingExtractionMetadata):
        payload["status"] = metadata.status
        payload["snapshot_present"] = metadata.snapshot_present
    return payload


def _serialize_claim_previews(claims: list[ReExtractionClaimPreview]) -> list[dict[str, object]]:
    return [
        {
            "claim_id": str(claim.claim_id) if claim.claim_id is not None else None,
            "text": claim.text,
            "normalized_text": claim.normalized_text,
            "claim_type": claim.claim_type.value,
            "sequence_order": claim.sequence_order,
        }
        for claim in claims
    ]


def _build_preview_snapshot_item(
    assertion: object,
    comparison: AssertionReExtractionComparison,
) -> dict[str, object]:
    return {
        "assertion_id": str(assertion.id),
        "paragraph_index": assertion.paragraph_index,
        "sentence_index": assertion.sentence_index,
        "assertion_text": assertion.raw_text,
        "status": _preview_status_for_comparison(comparison),
        "existing_metadata": _build_metadata_snapshot(comparison.existing_metadata),
        "proposed_metadata": _build_metadata_snapshot(comparison.proposed_metadata),
        "materially_changed": comparison.materially_changed,
        "apply_requires_replacement": comparison.apply_requires_replacement,
        "can_apply": comparison.can_apply,
        "blocked_reasons": list(comparison.blocked_reasons),
        "existing_claims": _serialize_claim_previews(comparison.existing_claims),
        "proposed_claims": _serialize_claim_previews(comparison.proposed_claims),
    }


def _build_apply_snapshot_item(
    *,
    assertion: object,
    comparison: AssertionReExtractionComparison,
    status: BatchApplyStatus,
    claims_replaced: bool,
    metadata_updated: bool,
    resulting_claims: list[ClaimUnit],
    notes: list[str],
) -> dict[str, object]:
    return {
        "assertion_id": str(assertion.id),
        "paragraph_index": assertion.paragraph_index,
        "sentence_index": assertion.sentence_index,
        "assertion_text": assertion.raw_text,
        "status": status,
        "existing_metadata": _build_metadata_snapshot(comparison.existing_metadata),
        "proposed_metadata": _build_metadata_snapshot(comparison.proposed_metadata),
        "materially_changed": comparison.materially_changed,
        "apply_requires_replacement": comparison.apply_requires_replacement,
        "can_apply": comparison.can_apply,
        "claims_replaced": claims_replaced,
        "metadata_updated": metadata_updated,
        "blocked_reasons": list(comparison.blocked_reasons),
        "existing_claims": _serialize_claim_previews(comparison.existing_claims),
        "proposed_claims": _serialize_claim_previews(comparison.proposed_claims),
        "resulting_claims": _serialize_claim_previews([_build_claim_preview(claim) for claim in resulting_claims]),
        "notes": list(notes),
    }


def _build_draft_run_snapshot(
    *,
    draft_id: UUID,
    run_kind: ReExtractionRunKind,
    requested_mode: ExtractionMode,
    items: list[dict[str, object]],
    summary: dict[str, int],
) -> dict[str, object]:
    return {
        "draft_id": str(draft_id),
        "run_kind": run_kind.value,
        "requested_mode": requested_mode,
        "extraction_version": CURRENT_EXTRACTION_VERSION,
        "summary": summary,
        "items": items,
    }
