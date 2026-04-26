from uuid import uuid4

from app.core.enums import DecisionAction, SupportStatus
from app.models import UserDecision, VerificationRun
from app.services.audit.report_builder import build_audit_summary, build_draft_compile_summary
from app.services.llm.base import ProviderSupportAssessment
from app.services.verification.classifier import VerificationResult
from app.services.workflows.draft_compile import DraftCompileFlaggedClaim, DraftCompileResult, DraftCompileVerdictCounts


def build_run(
    *,
    verdict: SupportStatus = SupportStatus.PARTIALLY_SUPPORTED,
    flags: list[str] | None = None,
    support_snapshot_version: int | None = None,
    support_snapshot: dict[str, object] | None = None,
) -> VerificationRun:
    return VerificationRun(
        claim_unit_id=uuid4(),
        model_version="verification-stub-v1",
        prompt_version="prompt-v1",
        deterministic_flags=flags or [],
        verdict=verdict,
        reasoning="Persisted reasoning.",
        support_snapshot_version=support_snapshot_version,
        support_snapshot=support_snapshot,
        suggested_fix=None,
        confidence_score=0.6,
    )


def build_result(
    *,
    verdict: SupportStatus = SupportStatus.PARTIALLY_SUPPORTED,
    flags: list[str] | None = None,
    primary_anchor: str | None = None,
    support_assessments: list[ProviderSupportAssessment] | None = None,
) -> VerificationResult:
    return VerificationResult(
        model_version="verification-stub-v1",
        prompt_version="prompt-v1",
        deterministic_flags=flags or [],
        verdict=verdict,
        reasoning="Fresh verification reasoning.",
        suggested_fix=None,
        confidence_score=0.6,
        primary_anchor=primary_anchor,
        support_assessments=support_assessments or [],
    )


def build_decision(action: DecisionAction, note: str | None = None) -> UserDecision:
    return UserDecision(
        verification_run_id=uuid4(),
        action=action,
        note=note,
    )


def build_assessment(anchor: str, *, role: str, contribution: str) -> ProviderSupportAssessment:
    return ProviderSupportAssessment(
        segment_id=uuid4(),
        anchor=anchor,
        role=role,
        contribution=contribution,
    )


def build_support_snapshot(
    *,
    scope_kind: str = "bundle",
    primary_anchor: str | None = None,
    support_assessments: list[dict[str, object]] | None = None,
    excluded_support_links: list[dict[str, object]] | None = None,
    allowed_source_document_ids: list[str] | None = None,
) -> dict[str, object]:
    return {
        "claim_scope": {
            "claim_id": str(uuid4()),
            "draft_id": str(uuid4()),
            "matter_id": str(uuid4()),
            "evidence_bundle_id": str(uuid4()) if scope_kind == "bundle" else None,
            "scope_kind": scope_kind,
            "allowed_source_document_ids": allowed_source_document_ids or [str(uuid4())],
        },
        "valid_support_links": [],
        "excluded_support_links": excluded_support_links or [],
        "support_items": [],
        "citations": [],
        "provider_output": {
            "primary_anchor": primary_anchor,
            "support_assessments": support_assessments or [],
        },
    }


def build_compile_result() -> DraftCompileResult:
    return DraftCompileResult(
        draft_id=uuid4(),
        total_claims=4,
        counts=DraftCompileVerdictCounts(
            supported=1,
            partially_supported=0,
            overstated=1,
            ambiguous=1,
            unsupported=1,
            unverified=0,
        ),
        flagged_claims=[
            DraftCompileFlaggedClaim(
                claim_id=uuid4(),
                claim_text="Doe reviewed the contract and approved the invoice.",
                verdict=SupportStatus.OVERSTATED,
                deterministic_flags=["narrow_support"],
                primary_anchor="p.12:3-12:4",
                support_assessments=[],
                suggested_fix="Split the claim.",
                confidence_score=0.55,
            ),
            DraftCompileFlaggedClaim(
                claim_id=uuid4(),
                claim_text="Doe signed the contract.",
                verdict=SupportStatus.AMBIGUOUS,
                deterministic_flags=["contextual_support_only"],
                primary_anchor="p.13:1-13:2",
                support_assessments=[],
                suggested_fix="Add direct answer testimony.",
                confidence_score=0.4,
            ),
        ],
    )


def test_build_audit_summary_includes_primary_anchor_from_fresh_result() -> None:
    run = build_run(flags=["absolute_qualifier_mismatch"])
    result = build_result(
        flags=["absolute_qualifier_mismatch"],
        primary_anchor="p.10:3-10:4",
    )

    summary = build_audit_summary(run, [], verification_result=result)

    assert "Verdict: PARTIALLY_SUPPORTED." in summary
    assert "Flags: absolute_qualifier_mismatch." in summary
    assert "Primary support anchor: p.10:3-10:4." in summary


def test_build_audit_summary_prefers_persisted_support_snapshot() -> None:
    run = build_run(
        support_snapshot_version=1,
        support_snapshot=build_support_snapshot(
            primary_anchor="p.22:1-22:2",
            support_assessments=[
                {
                    "segment_id": str(uuid4()),
                    "anchor": "p.22:1-22:2",
                    "role": "primary",
                    "contribution": "Persisted snapshot support.",
                }
            ],
            excluded_support_links=[
                {
                    "link_id": str(uuid4()),
                    "claim_id": str(uuid4()),
                    "segment_id": str(uuid4()),
                    "code": "out_of_scope_support_link",
                    "message": "Segment source document is outside the draft evidence bundle.",
                }
            ],
            allowed_source_document_ids=[str(uuid4()), str(uuid4())],
        )
    )
    result = build_result(
        primary_anchor="p.99:1-99:2",
        support_assessments=[
            build_assessment(
                "p.99:1-99:2",
                role="primary",
                contribution="Fresh result that should be ignored for history.",
            )
        ],
    )

    summary = build_audit_summary(run, [], verification_result=result)

    assert "Scope: bundle (2 allowed source document(s))." in summary
    assert "Primary support anchor: p.22:1-22:2." in summary
    assert "Persisted snapshot support." in summary
    assert "Fresh result that should be ignored for history." not in summary
    assert "Excluded links: out_of_scope_support_link x1 (Segment source document is outside the draft evidence bundle.)." in summary


def test_build_audit_summary_marks_legacy_unversioned_snapshot_as_legacy() -> None:
    run = build_run(
        support_snapshot=build_support_snapshot(
            scope_kind="matter_fallback",
            primary_anchor="p.15:1-15:2",
        )
    )

    summary = build_audit_summary(run, [])

    assert "Snapshot: legacy/unversioned persisted support snapshot interpreted as v1-compatible." in summary
    assert "Scope: matter_fallback (1 allowed source document(s))." in summary
    assert "Primary support anchor: p.15:1-15:2." in summary


def test_build_audit_summary_handles_unsupported_snapshot_version_conservatively() -> None:
    run = build_run(
        support_snapshot_version=99,
        support_snapshot=build_support_snapshot(primary_anchor="p.44:1-44:2"),
        flags=["subject_mismatch"],
    )

    summary = build_audit_summary(run, [])

    assert "Snapshot: persisted support snapshot version 99 is not supported for structured rendering." in summary
    assert "Flags: subject_mismatch." in summary
    assert "Primary support anchor:" not in summary
    assert "Support items:" not in summary
    assert "Excluded links:" not in summary


def test_build_audit_summary_includes_support_assessments_in_order() -> None:
    run = build_run()
    result = build_result(
        primary_anchor="p.12:3-12:4",
        support_assessments=[
            build_assessment(
                "p.12:3-12:4",
                role="primary",
                contribution="Primary support with strong lexical overlap from answer testimony.",
            ),
            build_assessment(
                "p.12:1-12:2",
                role="contextual",
                contribution="Contextual support with no direct lexical overlap from question framing.",
            ),
        ],
    )

    summary = build_audit_summary(run, [], verification_result=result)

    assert "Support items:" in summary
    assert "p.12:3-12:4 [primary]: Primary support with strong lexical overlap from answer testimony." in summary
    assert "p.12:1-12:2 [contextual]: Contextual support with no direct lexical overlap from question framing." in summary
    assert summary.index("p.12:3-12:4 [primary]") < summary.index("p.12:1-12:2 [contextual]")


def test_build_audit_summary_falls_back_to_persisted_run_when_no_structured_support_data() -> None:
    run = build_run(
        verdict=SupportStatus.UNSUPPORTED,
        flags=["missing_citation"],
    )

    summary = build_audit_summary(run, [])

    assert summary == "Verdict: UNSUPPORTED. Flags: missing_citation. User decisions: none."
    assert "Primary support anchor:" not in summary
    assert "Support items:" not in summary
    assert "Excluded links:" not in summary


def test_build_audit_summary_includes_user_decisions_and_notes() -> None:
    run = build_run()
    result = build_result(primary_anchor="p.14:6-14:7")
    decisions = [
        build_decision(DecisionAction.ACKNOWLEDGE_INFERENCE, note="Inference rather than direct testimony."),
        build_decision(DecisionAction.ESCALATE_FOR_REVIEW),
    ]

    summary = build_audit_summary(run, decisions, verification_result=result)

    assert "User decisions:" in summary
    assert "ACKNOWLEDGE_INFERENCE (Inference rather than direct testimony.)" in summary
    assert "ESCALATE_FOR_REVIEW" in summary


def test_build_draft_compile_summary_includes_counts_and_flagged_claims() -> None:
    result = build_compile_result()

    summary = build_draft_compile_summary(result)

    assert f"Draft {result.draft_id} compile summary." in summary
    assert "Total claims: 4." in summary
    assert "supported=1" in summary
    assert "overstated=1" in summary
    assert "ambiguous=1" in summary
    assert "unsupported=1" in summary
    assert "OVERSTATED" in summary
    assert "AMBIGUOUS" in summary
    assert "p.12:3-12:4" in summary
    assert "narrow_support" in summary
