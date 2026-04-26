"""Draft routes."""

from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.core.exceptions import NotFoundError
from app.repositories.claims import ClaimsRepository
from app.repositories.drafts import DraftRepository
from app.repositories.matters import MatterRepository
from app.repositories.review_decisions import ClaimReviewDecisionRepository
from app.schemas.citation_verification import (
    CitationAuthorityStatusCountsRead,
    CitationMatchedAuthorityRead,
    CitationParsedAuthorityRead,
    CitationVerificationItemRead,
    CitationVerificationRequest,
    CitationVerificationResultRead,
    CitationVerificationSummaryRead,
    CitationVerdictCountsRead,
)
from app.schemas.draft import DraftCreate, DraftRead, DraftTextCreate, DraftTextCreateRead
from app.schemas.draft_workflow import (
    DraftClaimChangeSummaryRead,
    DraftClaimRelationshipRead,
    DraftCompileResultRead,
    DraftExcludedLinkRead,
    DraftFlaggedClaimCountsRead,
    DraftFlaggedClaimRead,
    DraftReviewFlagBucketRead,
    DraftReviewDecisionSummaryRead,
    DraftReviewFreshnessRead,
    DraftReviewIntelligenceSummaryRead,
    DraftReviewIssueBucketsRead,
    DraftReviewOverviewRead,
    DraftReviewRunSummaryRead,
    DraftResolvedFlaggedClaimRead,
    DraftReviewResultRead,
    DraftReviewScopeRead,
    DraftWeakSupportClusterRead,
    DraftVerdictCountsRead,
)
from app.schemas.reextraction import (
    DraftReExtractionApplyRead,
    DraftReExtractionPreviewRead,
    ReExtractionApplyRequest,
    ReExtractionCompareRequest,
)
from app.schemas.review_decision import DraftReviewQueueStateRead
from app.schemas.verification import SupportAssessmentRead
from app.services.workflows.draft_compile import (
    DraftCompileFlaggedClaim,
    DraftCompileResult,
    DraftCompileService,
    DraftCompileVerdictCounts,
)
from app.services.workflows.draft_review import (
    DraftReviewDecisionSummary,
    DraftReviewFlagBucket,
    DraftReviewFlaggedClaimCounts,
    DraftReviewFreshness,
    DraftReviewIssueBuckets,
    DraftReviewOverview,
    DraftReviewExecutionService,
    DraftReviewReadService,
    DraftReviewRunSummary,
    DraftReviewResolvedClaim,
    DraftReviewResult,
    DraftReviewService,
)
from app.services.workflows.draft_intake import DraftIntakeService
from app.services.workflows.citation_verification import CitationVerificationResult, CitationVerificationService
from app.services.workflows.reextract_claims import (
    DraftReExtractionApplyItem,
    DraftReExtractionApplyResult,
    DraftReExtractionPreviewItem,
    DraftReExtractionPreviewResult,
    ClaimReExtractionService,
)

router = APIRouter(prefix="/api/v1", tags=["drafts"])


@router.post("/citation-verification", response_model=CitationVerificationResultRead, status_code=status.HTTP_201_CREATED)
def verify_citations(
    payload: CitationVerificationRequest,
    db: Session = Depends(get_db_session),
):
    result = CitationVerificationService(db).verify_draft_text(payload)
    db.commit()
    return _build_citation_verification_response(result)


@router.post("/drafts", response_model=DraftTextCreateRead, status_code=status.HTTP_201_CREATED)
def create_draft_from_text(
    payload: DraftTextCreate,
    db: Session = Depends(get_db_session),
):
    result = DraftIntakeService(db).create_draft_from_text(payload)
    db.commit()
    return DraftTextCreateRead(
        draft_id=result.draft_id,
        matter_id=result.matter_id,
        title=result.title,
        assertion_count=result.assertion_count,
        claim_count=result.claim_count,
    )


@router.post("/matters/{matter_id}/drafts", response_model=DraftRead, status_code=status.HTTP_201_CREATED)
def create_draft(
    matter_id: UUID,
    payload: DraftCreate,
    db: Session = Depends(get_db_session),
):
    if MatterRepository(db).get(matter_id) is None:
        raise NotFoundError("Matter not found.")
    draft = DraftRepository(db).create(matter_id, payload)
    db.commit()
    db.refresh(draft)
    return draft


@router.get("/drafts/{draft_id}", response_model=DraftRead)
def get_draft(draft_id: UUID, db: Session = Depends(get_db_session)):
    draft = DraftRepository(db).get(draft_id)
    if draft is None:
        raise NotFoundError("Draft not found.")
    return draft


@router.post("/drafts/{draft_id}/compile", response_model=DraftCompileResultRead)
def compile_draft(draft_id: UUID, db: Session = Depends(get_db_session)):
    result = DraftCompileService(db).compile_draft(draft_id)
    db.commit()
    return _build_compile_response(result)


@router.post("/drafts/{draft_id}/review", response_model=DraftReviewResultRead)
def review_draft(draft_id: UUID, db: Session = Depends(get_db_session)):
    review_result = DraftReviewExecutionService(db).execute_review(draft_id)
    db.commit()
    return _build_review_response(review_result)


@router.get("/drafts/{draft_id}/review-state", response_model=DraftReviewResultRead)
def get_review_state(draft_id: UUID, db: Session = Depends(get_db_session)):
    result = DraftReviewReadService(db).read_review_state(draft_id)
    return _build_review_response(result)


@router.post("/drafts/{draft_id}/reextract/preview", response_model=DraftReExtractionPreviewRead)
def preview_draft_reextraction(
    draft_id: UUID,
    payload: ReExtractionCompareRequest | None = None,
    db: Session = Depends(get_db_session),
):
    result = ClaimReExtractionService(db).preview_draft(
        draft_id,
        mode=(payload.mode if payload is not None else "structured"),
    )
    db.commit()
    return _build_draft_reextraction_preview_response(result)


@router.post("/drafts/{draft_id}/reextract/apply", response_model=DraftReExtractionApplyRead)
def apply_draft_reextraction(
    draft_id: UUID,
    payload: ReExtractionApplyRequest | None = None,
    db: Session = Depends(get_db_session),
):
    result = ClaimReExtractionService(db).apply_draft(
        draft_id,
        mode=(payload.mode if payload is not None else "structured"),
    )
    db.commit()
    return _build_draft_reextraction_apply_response(result)


def _build_compile_response(result: DraftCompileResult) -> DraftCompileResultRead:
    return DraftCompileResultRead(
        draft_id=result.draft_id,
        total_claims=result.total_claims,
        verdict_counts=_build_verdict_counts_response(result.counts),
        flagged_claims=[_build_flagged_claim_response(claim) for claim in result.flagged_claims],
    )


def _build_citation_verification_response(
    result: CitationVerificationResult,
) -> CitationVerificationResultRead:
    return CitationVerificationResultRead(
        draft_id=result.draft_id,
        matter_id=result.matter_id,
        title=result.title,
        review_run_id=result.review_run_id,
        reviewed_at=result.reviewed_at,
        summary=CitationVerificationSummaryRead(
            total_claims=result.summary.total_claims,
            total_cited_propositions=result.summary.total_cited_propositions,
            flagged_citation_count=result.summary.flagged_citation_count,
            verdict_counts=CitationVerdictCountsRead(
                supported=result.summary.verdict_counts.supported,
                partially_supported=result.summary.verdict_counts.partially_supported,
                overstated=result.summary.verdict_counts.overstated,
                ambiguous=result.summary.verdict_counts.ambiguous,
                unsupported=result.summary.verdict_counts.unsupported,
                unverified=result.summary.verdict_counts.unverified,
                contradicted=result.summary.verdict_counts.contradicted,
            ),
            authority_status_counts=CitationAuthorityStatusCountsRead(
                authority_unverified=result.summary.authority_status_counts.authority_unverified,
                citation_recognized=result.summary.authority_status_counts.citation_recognized,
                authority_candidate_parsed=result.summary.authority_status_counts.authority_candidate_parsed,
                authority_matched=result.summary.authority_status_counts.authority_matched,
                linked_authority_support_present=(
                    result.summary.authority_status_counts.linked_authority_support_present
                ),
                not_reviewed=result.summary.authority_status_counts.not_reviewed,
            ),
        ),
        citations=[
            CitationVerificationItemRead(
                claim_id=item.claim_id,
                draft_sequence=item.draft_sequence,
                citation_text=item.citation_text,
                proposition_text=item.proposition_text,
                assertion_context=item.assertion_context,
                authority_status=item.authority_status,
                authority_match_status=item.authority_match_status,
                parsed_authority=(
                    CitationParsedAuthorityRead(
                        case_name=item.parsed_authority.case_name,
                        reporter_volume=item.parsed_authority.reporter_volume,
                        reporter_abbreviation=item.parsed_authority.reporter_abbreviation,
                        first_page=item.parsed_authority.first_page,
                        court=item.parsed_authority.court,
                        year=item.parsed_authority.year,
                    )
                    if item.parsed_authority is not None
                    else None
                ),
                normalized_authority_reference=item.normalized_authority_reference,
                matched_authority=(
                    CitationMatchedAuthorityRead(
                        authority_id=item.matched_authority.authority_id,
                        canonical_name=item.matched_authority.canonical_name,
                        canonical_citation=item.matched_authority.canonical_citation,
                        reporter_volume=item.matched_authority.reporter_volume,
                        reporter_abbreviation=item.matched_authority.reporter_abbreviation,
                        first_page=item.matched_authority.first_page,
                        court=item.matched_authority.court,
                        year=item.matched_authority.year,
                        source_name=item.matched_authority.source_name,
                    )
                    if item.matched_authority is not None
                    else None
                ),
                proposition_verdict=item.proposition_verdict,
                reasoning=item.reasoning,
                reasoning_categories=item.reasoning_categories,
                confidence_score=item.confidence_score,
                primary_anchor=item.primary_anchor,
                support_snippet=item.support_snippet,
                suggested_fix=item.suggested_fix,
                verification_run_id=item.verification_run_id,
                verified_at=item.verified_at,
            )
            for item in result.citations
        ],
    )


def _build_review_response(result: DraftReviewResult) -> DraftReviewResultRead:
    return DraftReviewResultRead(
        draft_id=result.draft_id,
        total_claims=result.total_claims,
        verdict_counts=_build_verdict_counts_response(result.verdict_counts),
        flagged_claim_counts=_build_flagged_claim_counts_response(result.flagged_claim_counts),
        review_overview=_build_review_overview_response(result.review_overview),
        freshness=_build_review_freshness_response(result.freshness),
        queue_state=_build_draft_review_queue_state_response(result.queue_state),
        active_queue_claims=[_build_flagged_claim_response(claim) for claim in result.active_queue_claims],
        resolved_claims=[_build_resolved_claim_response(claim) for claim in result.resolved_claims],
        latest_review_run=_build_review_run_summary_response(result.latest_review_run),
        previous_review_run=_build_review_run_summary_response(result.previous_review_run),
        intelligence_summary=_build_intelligence_summary_response(result.intelligence_summary),
        issue_buckets=_build_issue_buckets_response(result.issue_buckets),
        flag_buckets=[_build_flag_bucket_response(bucket) for bucket in result.flag_buckets],
        top_risky_claims=[_build_flagged_claim_response(claim) for claim in result.top_risky_claims],
        summary=result.summary,
    )


def _build_verdict_counts_response(counts: DraftCompileVerdictCounts) -> DraftVerdictCountsRead:
    return DraftVerdictCountsRead(
        supported=counts.supported,
        partially_supported=counts.partially_supported,
        overstated=counts.overstated,
        ambiguous=counts.ambiguous,
        unsupported=counts.unsupported,
        unverified=counts.unverified,
    )


def _build_flagged_claim_counts_response(
    counts: DraftReviewFlaggedClaimCounts,
) -> DraftFlaggedClaimCountsRead:
    return DraftFlaggedClaimCountsRead(
        unsupported=counts.unsupported,
        ambiguous=counts.ambiguous,
        overstated=counts.overstated,
        unverified=counts.unverified,
        total=counts.total,
    )


def _build_issue_buckets_response(buckets: DraftReviewIssueBuckets) -> DraftReviewIssueBucketsRead:
    return DraftReviewIssueBucketsRead(
        unsupported=[_build_flagged_claim_response(claim) for claim in buckets.unsupported],
        overstated=[_build_flagged_claim_response(claim) for claim in buckets.overstated],
        ambiguous=[_build_flagged_claim_response(claim) for claim in buckets.ambiguous],
        unverified=[_build_flagged_claim_response(claim) for claim in buckets.unverified],
    )


def _build_review_overview_response(overview: DraftReviewOverview) -> DraftReviewOverviewRead:
    return DraftReviewOverviewRead(
        total_claims=overview.total_claims,
        total_flagged_claims=overview.total_flagged_claims,
        highest_severity_bucket=overview.highest_severity_bucket,
        top_issue_categories=list(overview.top_issue_categories),
    )


def _build_review_freshness_response(freshness: DraftReviewFreshness) -> DraftReviewFreshnessRead:
    return DraftReviewFreshnessRead(
        state_source=freshness.state_source,
        has_persisted_review_runs=freshness.has_persisted_review_runs,
        last_review_run_at=freshness.last_review_run_at,
        latest_review_run_id=freshness.latest_review_run_id,
        latest_review_run_status=(
            freshness.latest_review_run_status.value if freshness.latest_review_run_status is not None else None
        ),
        latest_decision_at=freshness.latest_decision_at,
        has_decisions_after_latest_run=freshness.has_decisions_after_latest_run,
        latest_claim_verification_at=freshness.latest_claim_verification_at,
        latest_verification_run_id=freshness.latest_verification_run_id,
        has_verification_activity_after_latest_run=freshness.has_verification_activity_after_latest_run,
        is_stale=freshness.is_stale,
    )


def _build_draft_review_queue_state_response(queue_state: object) -> DraftReviewQueueStateRead:
    return DraftReviewQueueStateRead(
        draft_id=queue_state.draft_id,
        total_flagged_claims=queue_state.total_flagged_claims,
        resolved_flagged_claims=queue_state.resolved_flagged_claims,
        remaining_flagged_claims=queue_state.remaining_flagged_claims,
        next_claim_id=queue_state.next_claim_id,
    )


def _build_flag_bucket_response(bucket: DraftReviewFlagBucket) -> DraftReviewFlagBucketRead:
    return DraftReviewFlagBucketRead(
        flag=bucket.flag,
        claim_count=bucket.claim_count,
        claims=[_build_flagged_claim_response(claim) for claim in bucket.claims],
    )


def _build_review_decision_summary_response(
    summary: DraftReviewDecisionSummary,
) -> DraftReviewDecisionSummaryRead:
    return DraftReviewDecisionSummaryRead(
        action=summary.action,
        note=summary.note,
        proposed_replacement_text=summary.proposed_replacement_text,
        created_at=summary.created_at,
    )


def _build_resolved_claim_response(claim: DraftReviewResolvedClaim) -> DraftResolvedFlaggedClaimRead:
    return DraftResolvedFlaggedClaimRead(
        claim=_build_flagged_claim_response(claim.claim),
        latest_decision=_build_review_decision_summary_response(claim.latest_decision),
    )


def _build_review_run_summary_response(
    summary: DraftReviewRunSummary | None,
) -> DraftReviewRunSummaryRead | None:
    if summary is None:
        return None
    return DraftReviewRunSummaryRead(
        run_id=summary.run_id,
        status=summary.status.value,
        created_at=summary.created_at,
        total_claims=summary.total_claims,
        total_flagged_claims=summary.total_flagged_claims,
        resolved_flagged_claims=summary.resolved_flagged_claims,
        remaining_flagged_claims=summary.remaining_flagged_claims,
        highest_severity_bucket=summary.highest_severity_bucket,
    )


def _build_flagged_claim_response(claim: DraftCompileFlaggedClaim) -> DraftFlaggedClaimRead:
    return DraftFlaggedClaimRead(
        claim_id=claim.claim_id,
        draft_sequence=claim.draft_sequence,
        claim_text=claim.claim_text,
        verdict=claim.verdict,
        assertion_context=claim.assertion_context,
        reasoning=claim.reasoning,
        deterministic_flags=list(claim.deterministic_flags),
        primary_anchor=claim.primary_anchor,
        support_assessments=[
            SupportAssessmentRead(
                segment_id=assessment.segment_id,
                anchor=assessment.anchor,
                role=assessment.role,
                contribution=assessment.contribution,
            )
            for assessment in claim.support_assessments
        ],
        excluded_links=[
            DraftExcludedLinkRead(
                code=entry.get("code"),
                message=entry.get("message"),
            )
            for entry in claim.excluded_links
        ],
        scope=(
            DraftReviewScopeRead(
                scope_kind=str(claim.scope.get("scope_kind")),
                allowed_source_document_count=int(claim.scope.get("allowed_source_document_count", 0)),
            )
            if claim.scope is not None
            else None
        ),
        suggested_fix=claim.suggested_fix,
        confidence_score=claim.confidence_score,
        latest_verification_run_id=claim.latest_verification_run_id,
        latest_verification_run_at=claim.latest_verification_run_at,
        reasoning_categories=list(claim.reasoning_categories),
        changed_since_last_run=claim.changed_since_last_run,
        change_summary=_build_claim_change_summary_response(claim.change_summary),
        contradiction_flags=list(claim.contradiction_flags),
        claim_relationships=[
            DraftClaimRelationshipRead(
                relationship_type=relationship.relationship_type,
                related_claim_id=relationship.related_claim_id,
                related_claim_text=relationship.related_claim_text,
                reason_code=relationship.reason_code,
                reason_text=relationship.reason_text,
                confidence_score=relationship.confidence_score,
            )
            for relationship in claim.claim_relationships
        ],
    )


def _build_claim_change_summary_response(change_summary) -> DraftClaimChangeSummaryRead | None:
    if change_summary is None:
        return None
    return DraftClaimChangeSummaryRead(
        current_verdict=change_summary.current_verdict,
        previous_verdict=change_summary.previous_verdict,
        verdict_changed=change_summary.verdict_changed,
        current_confidence_score=change_summary.current_confidence_score,
        previous_confidence_score=change_summary.previous_confidence_score,
        confidence_changed=change_summary.confidence_changed,
        current_primary_anchor=change_summary.current_primary_anchor,
        previous_primary_anchor=change_summary.previous_primary_anchor,
        primary_anchor_changed=change_summary.primary_anchor_changed,
        support_changed=change_summary.support_changed,
        current_support_assessment_count=change_summary.current_support_assessment_count,
        previous_support_assessment_count=change_summary.previous_support_assessment_count,
        current_excluded_link_count=change_summary.current_excluded_link_count,
        previous_excluded_link_count=change_summary.previous_excluded_link_count,
        current_flags=list(change_summary.current_flags),
        previous_flags=list(change_summary.previous_flags),
        flags_changed=change_summary.flags_changed,
        current_reasoning_categories=list(change_summary.current_reasoning_categories),
        previous_reasoning_categories=list(change_summary.previous_reasoning_categories),
        reasoning_categories_changed=change_summary.reasoning_categories_changed,
        changed_since_last_run=change_summary.changed_since_last_run,
    )


def _build_intelligence_summary_response(summary) -> DraftReviewIntelligenceSummaryRead | None:
    if summary is None:
        return None
    return DraftReviewIntelligenceSummaryRead(
        risk_distribution=DraftVerdictCountsRead(
            supported=summary.risk_distribution["supported"],
            partially_supported=summary.risk_distribution["partially_supported"],
            overstated=summary.risk_distribution["overstated"],
            ambiguous=summary.risk_distribution["ambiguous"],
            unsupported=summary.risk_distribution["unsupported"],
            unverified=summary.risk_distribution["unverified"],
        ),
        most_unstable_claim_ids=list(summary.most_unstable_claim_ids),
        repeatedly_changed_claim_ids=list(summary.repeatedly_changed_claim_ids),
        weak_support_claim_ids=list(summary.weak_support_claim_ids),
        contradiction_claim_ids=list(summary.contradiction_claim_ids),
        contradiction_pair_count=summary.contradiction_pair_count,
        duplicate_pair_count=summary.duplicate_pair_count,
        weak_support_clusters=[
            DraftWeakSupportClusterRead(
                flag=cluster.flag,
                claim_count=cluster.claim_count,
                claim_ids=list(cluster.claim_ids),
            )
            for cluster in summary.weak_support_clusters
        ],
    )


def _build_draft_reextraction_preview_response(
    result: DraftReExtractionPreviewResult,
) -> DraftReExtractionPreviewRead:
    return DraftReExtractionPreviewRead(
        run_id=result.run_id,
        draft_id=result.draft_id,
        requested_mode=result.requested_mode,
        total_assertions=result.total_assertions,
        ready_assertions=result.ready_assertions,
        unchanged_assertions=result.unchanged_assertions,
        blocked_assertions=result.blocked_assertions,
        materially_changed_assertions=result.materially_changed_assertions,
        legacy_unversioned_assertions=result.legacy_unversioned_assertions,
        items=[_build_draft_reextraction_preview_item_response(item) for item in result.items],
    )


def _build_draft_reextraction_preview_item_response(
    item: DraftReExtractionPreviewItem,
) -> dict[str, object]:
    return {
        "assertion_id": item.assertion_id,
        "paragraph_index": item.paragraph_index,
        "sentence_index": item.sentence_index,
        "assertion_text": item.assertion_text,
        "status": item.status,
        "existing_metadata": _build_existing_extraction_metadata_response(item.existing_metadata),
        "proposed_metadata": _build_proposed_extraction_metadata_response(item.proposed_metadata),
        "materially_changed": item.materially_changed,
        "apply_requires_replacement": item.apply_requires_replacement,
        "can_apply": item.can_apply,
        "blocked_reasons": list(item.blocked_reasons),
        "existing_claim_count": item.existing_claim_count,
        "proposed_claim_count": item.proposed_claim_count,
    }


def _build_draft_reextraction_apply_response(
    result: DraftReExtractionApplyResult,
) -> DraftReExtractionApplyRead:
    return DraftReExtractionApplyRead(
        run_id=result.run_id,
        draft_id=result.draft_id,
        requested_mode=result.requested_mode,
        total_assertions=result.total_assertions,
        applied_assertions=result.applied_assertions,
        skipped_assertions=result.skipped_assertions,
        blocked_assertions=result.blocked_assertions,
        replaced_assertions=result.replaced_assertions,
        metadata_only_assertions=result.metadata_only_assertions,
        items=[_build_draft_reextraction_apply_item_response(item) for item in result.items],
    )


def _build_draft_reextraction_apply_item_response(
    item: DraftReExtractionApplyItem,
) -> dict[str, object]:
    return {
        "assertion_id": item.assertion_id,
        "paragraph_index": item.paragraph_index,
        "sentence_index": item.sentence_index,
        "assertion_text": item.assertion_text,
        "status": item.status,
        "existing_metadata": _build_existing_extraction_metadata_response(item.existing_metadata),
        "proposed_metadata": _build_proposed_extraction_metadata_response(item.proposed_metadata),
        "materially_changed": item.materially_changed,
        "apply_requires_replacement": item.apply_requires_replacement,
        "can_apply": item.can_apply,
        "claims_replaced": item.claims_replaced,
        "metadata_updated": item.metadata_updated,
        "blocked_reasons": list(item.blocked_reasons),
        "resulting_claims": [_build_reextraction_claim_preview_response(claim) for claim in item.resulting_claims],
        "notes": list(item.notes),
    }


def _build_existing_extraction_metadata_response(metadata: object) -> dict[str, object]:
    return {
        "status": metadata.status,
        "strategy": metadata.strategy,
        "version": metadata.version,
        "snapshot_present": metadata.snapshot_present,
    }


def _build_proposed_extraction_metadata_response(metadata: object) -> dict[str, object]:
    return {
        "strategy": metadata.strategy,
        "version": metadata.version,
    }


def _build_reextraction_claim_preview_response(claim: object) -> dict[str, object]:
    return {
        "claim_id": claim.claim_id,
        "text": claim.text,
        "normalized_text": claim.normalized_text,
        "claim_type": claim.claim_type,
        "sequence_order": claim.sequence_order,
    }
