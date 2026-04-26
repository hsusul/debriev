from uuid import uuid4

from app.core.enums import ClaimReviewAction, SupportStatus
from app.models import ClaimReviewDecision
from app.services.audit.report_builder import build_draft_review_summary
from app.services.llm.base import ProviderSupportAssessment
from app.services.workflows.draft_compile import DraftCompileFlaggedClaim, DraftCompileResult, DraftCompileVerdictCounts
from app.services.workflows.draft_review import DraftReviewFreshness, DraftReviewService


def build_assessment(anchor: str, *, role: str, contribution: str) -> ProviderSupportAssessment:
    return ProviderSupportAssessment(
        segment_id=uuid4(),
        anchor=anchor,
        role=role,
        contribution=contribution,
    )


def build_compile_result() -> DraftCompileResult:
    ambiguous_one = DraftCompileFlaggedClaim(
        claim_id=uuid4(),
        claim_text="Doe signed the contract.",
        verdict=SupportStatus.AMBIGUOUS,
        deterministic_flags=["contextual_support_only"],
        primary_anchor="p.10:1-10:2",
        support_assessments=[
            build_assessment(
                "p.10:1-10:2",
                role="contextual",
                contribution="Question framing overlaps the claim more than direct answer testimony.",
            )
        ],
        suggested_fix="Add direct answer testimony.",
        confidence_score=0.4,
    )
    unsupported_one = DraftCompileFlaggedClaim(
        claim_id=uuid4(),
        claim_text="Doe delivered the notice.",
        verdict=SupportStatus.UNSUPPORTED,
        deterministic_flags=["subject_mismatch", "temporal_scope_mismatch"],
        primary_anchor="p.11:3-11:4",
        support_assessments=[
            build_assessment(
                "p.11:3-11:4",
                role="primary",
                contribution="Linked testimony discusses a different actor on a different date.",
            )
        ],
        suggested_fix="Relink the claim to matching testimony.",
        confidence_score=0.25,
    )
    overstated_one = DraftCompileFlaggedClaim(
        claim_id=uuid4(),
        claim_text="Doe reviewed the contract and approved the invoice.",
        verdict=SupportStatus.OVERSTATED,
        deterministic_flags=["narrow_support"],
        primary_anchor="p.12:5-12:6",
        support_assessments=[
            build_assessment(
                "p.12:5-12:6",
                role="primary",
                contribution="Covers only the review portion of the claim.",
            )
        ],
        suggested_fix="Split the claim into narrower propositions.",
        confidence_score=0.55,
    )
    unverified_one = DraftCompileFlaggedClaim(
        claim_id=uuid4(),
        claim_text="Doe admitted the error.",
        verdict=SupportStatus.UNVERIFIED,
        deterministic_flags=["missing_citation"],
        primary_anchor=None,
        support_assessments=[],
        suggested_fix="Attach an anchored segment before verification.",
        confidence_score=0.15,
    )
    unsupported_two = DraftCompileFlaggedClaim(
        claim_id=uuid4(),
        claim_text="Doe mailed the notice.",
        verdict=SupportStatus.UNSUPPORTED,
        deterministic_flags=["subject_mismatch"],
        primary_anchor="p.13:1-13:2",
        support_assessments=[],
        suggested_fix="Find testimony about mailing rather than signature review.",
        confidence_score=0.35,
    )
    overstated_two = DraftCompileFlaggedClaim(
        claim_id=uuid4(),
        claim_text="Smith always approved invoices.",
        verdict=SupportStatus.OVERSTATED,
        deterministic_flags=["absolute_qualifier_mismatch", "temporal_scope_mismatch"],
        primary_anchor="p.14:1-14:2",
        support_assessments=[],
        suggested_fix="Soften the qualifier.",
        confidence_score=0.4,
    )
    ambiguous_two = DraftCompileFlaggedClaim(
        claim_id=uuid4(),
        claim_text="Doe signed the amendment.",
        verdict=SupportStatus.AMBIGUOUS,
        deterministic_flags=["contextual_support_only"],
        primary_anchor="p.15:1-15:2",
        support_assessments=[],
        suggested_fix="Add answer testimony that states the fact directly.",
        confidence_score=0.4,
    )

    return DraftCompileResult(
        draft_id=uuid4(),
        total_claims=9,
        counts=DraftCompileVerdictCounts(
            supported=1,
            partially_supported=1,
            overstated=2,
            ambiguous=2,
            unsupported=2,
            unverified=1,
        ),
        flagged_claims=[
            ambiguous_one,
            unsupported_one,
            overstated_one,
            unverified_one,
            unsupported_two,
            overstated_two,
            ambiguous_two,
        ],
    )


def test_review_workflow_groups_flagged_claims_by_verdict_and_preserves_detail() -> None:
    compile_result = build_compile_result()

    review_result = DraftReviewService().review_draft(compile_result)

    assert review_result.draft_id == compile_result.draft_id
    assert review_result.total_claims == 9
    assert review_result.flagged_claim_counts.unsupported == 2
    assert review_result.flagged_claim_counts.overstated == 2
    assert review_result.flagged_claim_counts.ambiguous == 2
    assert review_result.flagged_claim_counts.unverified == 1
    assert review_result.review_overview.total_claims == 9
    assert review_result.review_overview.total_flagged_claims == 7
    assert review_result.review_overview.highest_severity_bucket == SupportStatus.UNSUPPORTED
    assert review_result.freshness == DraftReviewFreshness(
        state_source="persisted_read",
        has_persisted_review_runs=False,
        last_review_run_at=None,
        latest_review_run_id=None,
        latest_review_run_status=None,
        latest_decision_at=None,
        has_decisions_after_latest_run=False,
        latest_claim_verification_at=None,
        latest_verification_run_id=None,
        has_verification_activity_after_latest_run=False,
        is_stale=True,
    )
    assert review_result.review_overview.top_issue_categories == [
        "contextual_support_only",
        "subject_mismatch",
        "temporal_scope_mismatch",
    ]
    assert [claim.claim_text for claim in review_result.issue_buckets.unsupported] == [
        "Doe delivered the notice.",
        "Doe mailed the notice.",
    ]
    assert [claim.claim_text for claim in review_result.issue_buckets.overstated] == [
        "Doe reviewed the contract and approved the invoice.",
        "Smith always approved invoices.",
    ]
    assert [claim.claim_text for claim in review_result.issue_buckets.ambiguous] == [
        "Doe signed the contract.",
        "Doe signed the amendment.",
    ]
    assert [claim.claim_text for claim in review_result.issue_buckets.unverified] == [
        "Doe admitted the error.",
    ]
    preserved_claim = review_result.issue_buckets.unsupported[0]
    assert preserved_claim.primary_anchor == "p.11:3-11:4"
    assert preserved_claim.support_assessments[0].anchor == "p.11:3-11:4"
    assert preserved_claim.support_assessments[0].role == "primary"
    assert review_result.queue_state.total_flagged_claims == 7
    assert review_result.queue_state.resolved_flagged_claims == 0
    assert review_result.queue_state.remaining_flagged_claims == 7
    assert review_result.queue_state.next_claim_id == compile_result.flagged_claims[0].claim_id
    assert [claim.claim_text for claim in review_result.active_queue_claims] == [
        "Doe signed the contract.",
        "Doe delivered the notice.",
        "Doe reviewed the contract and approved the invoice.",
        "Doe admitted the error.",
        "Doe mailed the notice.",
        "Smith always approved invoices.",
        "Doe signed the amendment.",
    ]
    assert review_result.resolved_claims == []


def test_review_workflow_ranks_top_risky_claims_stably() -> None:
    compile_result = build_compile_result()

    review_result = DraftReviewService().review_draft(compile_result)

    assert [claim.claim_text for claim in review_result.top_risky_claims] == [
        "Doe delivered the notice.",
        "Doe mailed the notice.",
        "Smith always approved invoices.",
        "Doe reviewed the contract and approved the invoice.",
        "Doe signed the contract.",
    ]


def test_review_workflow_builds_flag_buckets_and_compact_summary() -> None:
    compile_result = build_compile_result()

    review_result = DraftReviewService().review_draft(compile_result)

    assert [bucket.flag for bucket in review_result.flag_buckets] == [
        "contextual_support_only",
        "subject_mismatch",
        "temporal_scope_mismatch",
        "absolute_qualifier_mismatch",
        "missing_citation",
        "narrow_support",
    ]
    assert [claim.claim_text for claim in review_result.flag_buckets[1].claims] == [
        "Doe delivered the notice.",
        "Doe mailed the notice.",
    ]
    assert [claim.claim_text for claim in review_result.flag_buckets[0].claims] == [
        "Doe signed the contract.",
        "Doe signed the amendment.",
    ]
    assert f"Draft {compile_result.draft_id} review summary." in review_result.summary
    assert "Flagged claims: 7." in review_result.summary
    assert "Highest severity: unsupported." in review_result.summary
    assert "Top issue categories: contextual_support_only, subject_mismatch, temporal_scope_mismatch." in review_result.summary
    assert "unsupported=2" in review_result.summary
    assert "overstated=2" in review_result.summary
    assert "Top review targets:" in review_result.summary
    assert "UNSUPPORTED" in review_result.summary


def test_report_builder_renders_review_summary_from_review_result() -> None:
    compile_result = build_compile_result()
    review_result = DraftReviewService().review_draft(compile_result)

    summary = build_draft_review_summary(review_result)

    assert summary == review_result.summary
    assert "Common flags:" in summary
    assert "subject_mismatch=2" in summary
    assert "Highest severity: unsupported." in summary


def test_review_workflow_projects_latest_decisions_out_of_active_queue() -> None:
    compile_result = build_compile_result()
    decided_claim = compile_result.flagged_claims[1]

    review_result = DraftReviewService().review_draft(
        compile_result,
        latest_decisions_by_claim={
            decided_claim.claim_id: ClaimReviewDecision(
                claim_unit_id=decided_claim.claim_id,
                draft_id=compile_result.draft_id,
                verification_run_id=None,
                action=ClaimReviewAction.MARK_FOR_REVISION,
                note="Split the actor/date mismatch before final review.",
                proposed_replacement_text=None,
            )
        },
    )

    assert review_result.queue_state.total_flagged_claims == 7
    assert review_result.queue_state.resolved_flagged_claims == 1
    assert review_result.queue_state.remaining_flagged_claims == 6
    assert [claim.claim_text for claim in review_result.active_queue_claims] == [
        "Doe signed the contract.",
        "Doe reviewed the contract and approved the invoice.",
        "Doe admitted the error.",
        "Doe mailed the notice.",
        "Smith always approved invoices.",
        "Doe signed the amendment.",
    ]
    assert review_result.resolved_claims[0].claim.claim_text == "Doe delivered the notice."
    assert review_result.resolved_claims[0].latest_decision.action == ClaimReviewAction.MARK_FOR_REVISION
    assert review_result.flagged_claim_counts.total == 6
