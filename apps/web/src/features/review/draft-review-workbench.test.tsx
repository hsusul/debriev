import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { DraftReviewWorkbench } from "./draft-review-workbench"

describe("DraftReviewWorkbench", () => {
  beforeEach(() => {
    window.history.replaceState({}, "", "/?draftId=draft-1")
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("hydrates from persisted review-state and renders freshness/history", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const method = init?.method ?? "GET"
      const url = String(input)

      if (url.endsWith("/api/v1/drafts/draft-1/review-state") && method === "GET") {
        return jsonResponse(
          buildReviewStatePayload({
            freshness: {
              has_persisted_review_runs: true,
              last_review_run_at: "2026-04-06T12:00:00Z",
              latest_review_run_id: "review-run-1",
              latest_review_run_status: "COMPLETED",
              latest_decision_at: null,
              has_decisions_after_latest_run: false,
              latest_claim_verification_at: "2026-04-06T11:58:00Z",
              latest_verification_run_id: "verification-run-1",
              has_verification_activity_after_latest_run: false,
              is_stale: false,
            },
            active_queue_claims: [
              buildFlaggedClaim({
                claim_id: "claim-1",
                claim_text: "Doe delivered the notice.",
                verdict: "unsupported",
                reasoning: "Linked testimony is about a different actor.",
                primary_anchor: "p.12:3-12:4",
                latest_verification_run_id: "verification-run-1",
                latest_verification_run_at: "2026-04-06T11:58:00Z",
              }),
            ],
            latest_review_run: buildReviewRun("review-run-1", "2026-04-06T12:00:00Z"),
          }),
        )
      }

      if (url.endsWith("/api/v1/claims/claim-1/review-history") && method === "GET") {
        return jsonResponse(buildClaimHistoryPayload("claim-1", "Doe delivered the notice."))
      }

      throw new Error(`Unhandled request: ${method} ${url}`)
    })

    vi.stubGlobal("fetch", fetchMock)

    render(<DraftReviewWorkbench />)

    expect(await screen.findAllByText("Doe delivered the notice.")).not.toHaveLength(0)
    expect(screen.getByText(/Last fresh run/i)).toBeInTheDocument()
    expect(screen.getByText(/1 unstable/i)).toBeInTheDocument()
    expect(await screen.findByText(/Verification changed from/i)).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/api/v1/drafts/draft-1/review-state",
      expect.objectContaining({
        headers: expect.objectContaining({ "Content-Type": "application/json" }),
      }),
    )
    expect(
      fetchMock.mock.calls.some(
        ([input, init]) => String(input).endsWith("/api/v1/drafts/draft-1/review") && init?.method === "POST",
      ),
    ).toBe(false)
  })

  it("submits a decision, refreshes from server truth, and keeps resolved claims inspectable", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const method = init?.method ?? "GET"
      const url = String(input)

      if (url.endsWith("/api/v1/drafts/draft-1/review-state") && method === "GET") {
        if (fetchMock.mock.calls.filter(([calledInput]) =>
          String(calledInput).endsWith("/api/v1/drafts/draft-1/review-state"),
        ).length === 1) {
          return jsonResponse(
            buildReviewStatePayload({
              freshness: {
                has_persisted_review_runs: true,
                last_review_run_at: "2026-04-06T12:00:00Z",
                latest_review_run_id: "review-run-1",
                latest_review_run_status: "COMPLETED",
                latest_decision_at: null,
                has_decisions_after_latest_run: false,
                latest_claim_verification_at: "2026-04-06T11:58:00Z",
                latest_verification_run_id: "verification-run-1",
                has_verification_activity_after_latest_run: false,
                is_stale: false,
              },
              active_queue_claims: [
                buildFlaggedClaim({
                  claim_id: "claim-1",
                  claim_text: "Doe delivered the notice.",
                  verdict: "unverified",
                }),
                buildFlaggedClaim({
                  claim_id: "claim-2",
                  claim_text: "Doe signed the contract.",
                  verdict: "ambiguous",
                  draft_sequence: 2,
                }),
              ],
              latest_review_run: buildReviewRun("review-run-1", "2026-04-06T12:00:00Z"),
            }),
          )
        }

        return jsonResponse(
          buildReviewStatePayload({
            freshness: {
              has_persisted_review_runs: true,
              last_review_run_at: "2026-04-06T12:00:00Z",
              latest_review_run_id: "review-run-1",
              latest_review_run_status: "COMPLETED",
              latest_decision_at: "2026-04-06T12:04:00Z",
              has_decisions_after_latest_run: true,
              latest_claim_verification_at: "2026-04-06T11:58:00Z",
              latest_verification_run_id: "verification-run-2",
              has_verification_activity_after_latest_run: false,
              is_stale: true,
            },
            active_queue_claims: [
              buildFlaggedClaim({
                claim_id: "claim-2",
                claim_text: "Doe signed the contract.",
                verdict: "ambiguous",
                draft_sequence: 2,
              }),
            ],
            resolved_claims: [
              {
                claim: buildFlaggedClaim({
                  claim_id: "claim-1",
                  claim_text: "Doe delivered the notice.",
                  verdict: "unverified",
                }),
                latest_decision: {
                  action: "acknowledge_risk",
                  note: null,
                  proposed_replacement_text: null,
                  created_at: "2026-04-06T12:04:00Z",
                },
              },
            ],
            latest_review_run: buildReviewRun("review-run-1", "2026-04-06T12:00:00Z"),
          }),
        )
      }

      if (url.endsWith("/api/v1/claims/claim-1/review-history") && method === "GET") {
        return jsonResponse(buildClaimHistoryPayload("claim-1", "Doe delivered the notice."))
      }

      if (url.endsWith("/api/v1/claims/claim-2/review-history") && method === "GET") {
        return jsonResponse(
          buildClaimHistoryPayload("claim-2", "Doe signed the contract.", {
            verdict: "ambiguous",
            primaryAnchor: "p.30:1-30:2",
          }),
        )
      }

      if (url.endsWith("/api/v1/claims/claim-1/decisions") && method === "POST") {
        return jsonResponse({
          decision: { id: "decision-1", claim_unit_id: "claim-1" },
          claim_review_state: { removed_from_active_queue: true },
          draft_queue: { next_claim_id: "claim-2" },
        })
      }

      throw new Error(`Unhandled request: ${method} ${url}`)
    })

    vi.stubGlobal("fetch", fetchMock)

    render(<DraftReviewWorkbench />)

    expect(await screen.findAllByText("Doe delivered the notice.")).not.toHaveLength(0)
    await userEvent.click(await screen.findByRole("button", { name: "Submit" }))

    expect(await screen.findAllByText("Doe signed the contract.")).not.toHaveLength(0)
    expect(await screen.findByText(/State changed after the last fresh run/i)).toBeInTheDocument()

    await userEvent.click(screen.getByRole("button", { name: /Doe delivered the notice\./i }))
    expect(await screen.findByText(/Resolved via/i)).toBeInTheDocument()
  })

  it("uses explicit rerun for fresh execution and keeps queue state stable across reload", async () => {
    let reviewStateCalls = 0
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const method = init?.method ?? "GET"
      const url = String(input)

      if (url.endsWith("/api/v1/drafts/draft-1/review-state") && method === "GET") {
        reviewStateCalls += 1
        if (reviewStateCalls === 1) {
          return jsonResponse(
            buildReviewStatePayload({
              freshness: {
                has_persisted_review_runs: false,
                last_review_run_at: null,
                latest_review_run_id: null,
                latest_review_run_status: null,
                latest_decision_at: null,
                has_decisions_after_latest_run: false,
                latest_claim_verification_at: null,
                latest_verification_run_id: null,
                has_verification_activity_after_latest_run: false,
                is_stale: true,
              },
              active_queue_claims: [
                buildFlaggedClaim({
                  claim_id: "claim-1",
                  claim_text: "Doe delivered the notice.",
                  verdict: "unsupported",
                }),
              ],
              latest_review_run: null,
              previous_review_run: null,
            }),
          )
        }

        return jsonResponse(
          buildReviewStatePayload({
            freshness: {
              has_persisted_review_runs: true,
              last_review_run_at: "2026-04-06T13:00:00Z",
              latest_review_run_id: "review-run-2",
              latest_review_run_status: "COMPLETED",
              latest_decision_at: null,
              has_decisions_after_latest_run: false,
              latest_claim_verification_at: "2026-04-06T12:59:00Z",
              latest_verification_run_id: "verification-run-2",
              has_verification_activity_after_latest_run: false,
              is_stale: false,
            },
            active_queue_claims: [
              buildFlaggedClaim({
                claim_id: "claim-1",
                claim_text: "Doe delivered the notice.",
                verdict: "unsupported",
              }),
            ],
            latest_review_run: buildReviewRun("review-run-2", "2026-04-06T13:00:00Z"),
            previous_review_run: buildReviewRun("review-run-1", "2026-04-06T12:00:00Z"),
          }),
        )
      }

      if (url.endsWith("/api/v1/claims/claim-1/review-history") && method === "GET") {
        return jsonResponse(buildClaimHistoryPayload("claim-1", "Doe delivered the notice."))
      }

      if (url.endsWith("/api/v1/drafts/draft-1/review") && method === "POST") {
        return jsonResponse(
          buildReviewStatePayload({
            freshness: {
              has_persisted_review_runs: true,
              last_review_run_at: "2026-04-06T13:00:00Z",
              latest_review_run_id: "review-run-2",
              latest_review_run_status: "COMPLETED",
              latest_decision_at: null,
              has_decisions_after_latest_run: false,
              latest_claim_verification_at: "2026-04-06T12:59:00Z",
              latest_verification_run_id: "verification-run-2",
              has_verification_activity_after_latest_run: false,
              is_stale: false,
            },
            active_queue_claims: [
              buildFlaggedClaim({
                claim_id: "claim-1",
                claim_text: "Doe delivered the notice.",
                verdict: "unsupported",
              }),
            ],
            latest_review_run: buildReviewRun("review-run-2", "2026-04-06T13:00:00Z"),
            previous_review_run: buildReviewRun("review-run-1", "2026-04-06T12:00:00Z"),
          }),
        )
      }

      throw new Error(`Unhandled request: ${method} ${url}`)
    })

    vi.stubGlobal("fetch", fetchMock)

    const firstRender = render(<DraftReviewWorkbench />)
    expect(await screen.findByText(/No persisted review run yet/i)).toBeInTheDocument()
    await userEvent.click(screen.getByRole("button", { name: "Run" }))

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(
          ([input, init]) => String(input).endsWith("/api/v1/drafts/draft-1/review") && init?.method === "POST",
        ),
      ).toBe(true),
    )

    firstRender.unmount()

    render(<DraftReviewWorkbench />)
    expect(await screen.findByText(/Last fresh run/i)).toBeInTheDocument()
    expect(await screen.findAllByText("Doe delivered the notice.")).not.toHaveLength(0)
  })
})

function jsonResponse(payload: unknown) {
  return Promise.resolve(
    new Response(JSON.stringify(payload), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }),
  )
}

function buildReviewRun(runId: string, createdAt: string) {
  return {
    run_id: runId,
    status: "COMPLETED",
    created_at: createdAt,
    total_claims: 3,
    total_flagged_claims: 1,
    resolved_flagged_claims: 0,
    remaining_flagged_claims: 1,
    highest_severity_bucket: "unsupported",
  }
}

function buildFlaggedClaim({
  claim_id,
  claim_text,
  verdict,
  draft_sequence = 1,
  reasoning = "Linked testimony is about a different actor.",
  primary_anchor = "p.12:3-12:4",
  latest_verification_run_id = "verification-run-1",
  latest_verification_run_at = "2026-04-06T11:58:00Z",
}: {
  claim_id: string
  claim_text: string
  verdict: string
  draft_sequence?: number
  reasoning?: string
  primary_anchor?: string | null
  latest_verification_run_id?: string | null
  latest_verification_run_at?: string | null
}) {
  return {
    claim_id,
    draft_sequence,
    claim_text,
    verdict,
    assertion_context: `${claim_text} Context.`,
    reasoning,
    deterministic_flags: ["subject_mismatch"],
    primary_anchor,
    support_assessments: [
      {
        segment_id: "segment-1",
        anchor: primary_anchor ?? "",
        role: "primary",
        contribution: "Direct support for the proposition.",
      },
    ],
    excluded_links: [],
    scope: {
      scope_kind: "bundle",
      allowed_source_document_count: 2,
    },
    suggested_fix: "Relink or revise the actor.",
    confidence_score: 0.2,
    latest_verification_run_id,
    latest_verification_run_at,
    reasoning_categories: ["scope_mismatch"],
    changed_since_last_run: true,
    change_summary: {
      current_verdict: verdict,
      previous_verdict: "ambiguous",
      verdict_changed: true,
      current_confidence_score: 0.2,
      previous_confidence_score: 0.4,
      confidence_changed: true,
      current_primary_anchor: primary_anchor,
      previous_primary_anchor: "p.10:1-10:2",
      primary_anchor_changed: true,
      support_changed: true,
      current_support_assessment_count: 1,
      previous_support_assessment_count: 0,
      current_excluded_link_count: 0,
      previous_excluded_link_count: 0,
      current_flags: ["subject_mismatch"],
      previous_flags: ["contextual_support_only"],
      flags_changed: true,
      current_reasoning_categories: ["scope_mismatch"],
      previous_reasoning_categories: ["weak_support"],
      reasoning_categories_changed: true,
      changed_since_last_run: true,
    },
    contradiction_flags: [],
    claim_relationships: [],
  }
}

function buildReviewStatePayload(overrides: Partial<Record<string, unknown>> = {}) {
  return {
    draft_id: "draft-1",
    total_claims: 3,
    flagged_claim_counts: { total: 1 },
    review_overview: {
      highest_severity_bucket: "unsupported",
      top_issue_categories: ["subject_mismatch"],
    },
    freshness: {
      state_source: "persisted_read",
      has_persisted_review_runs: true,
      last_review_run_at: "2026-04-06T12:00:00Z",
      latest_review_run_id: "review-run-1",
      latest_review_run_status: "COMPLETED",
      latest_decision_at: null,
      has_decisions_after_latest_run: false,
      latest_claim_verification_at: "2026-04-06T11:58:00Z",
      latest_verification_run_id: "verification-run-1",
      has_verification_activity_after_latest_run: false,
      is_stale: false,
    },
    queue_state: {
      draft_id: "draft-1",
      total_flagged_claims: 1,
      resolved_flagged_claims: 0,
      remaining_flagged_claims: 1,
      next_claim_id: "claim-1",
    },
    active_queue_claims: [buildFlaggedClaim({ claim_id: "claim-1", claim_text: "Doe delivered the notice.", verdict: "unsupported" })],
    resolved_claims: [],
    latest_review_run: buildReviewRun("review-run-1", "2026-04-06T12:00:00Z"),
    previous_review_run: null,
    intelligence_summary: {
      risk_distribution: {
        supported: 0,
        partially_supported: 0,
        overstated: 0,
        ambiguous: 0,
        unsupported: 1,
        unverified: 0,
      },
      most_unstable_claim_ids: ["claim-1"],
      repeatedly_changed_claim_ids: ["claim-1"],
      weak_support_claim_ids: [],
      contradiction_claim_ids: [],
      contradiction_pair_count: 0,
      duplicate_pair_count: 0,
      weak_support_clusters: [],
    },
    issue_buckets: {
      unsupported: [],
      overstated: [],
      ambiguous: [],
      unverified: [],
    },
    flag_buckets: [],
    top_risky_claims: [],
    summary: "Draft review summary.",
    ...overrides,
  }
}

function buildClaimHistoryPayload(
  claimId: string,
  claimText: string,
  overrides: { verdict?: string; primaryAnchor?: string; reviewDisposition?: "active" | "resolved" } = {},
) {
  const verdict = overrides.verdict ?? "unsupported"
  const primaryAnchor = overrides.primaryAnchor ?? "p.12:3-12:4"
  return {
    claim_id: claimId,
    draft_id: "draft-1",
    claim_text: claimText,
    assertion_context: `${claimText} Context.`,
    support_status: verdict,
    review_disposition: overrides.reviewDisposition ?? "active",
    latest_decision: null,
    decision_history: [],
    latest_verification: {
      id: `verification-${claimId}`,
      claim_unit_id: claimId,
      verdict,
      reasoning: "Linked testimony is about a different actor.",
      deterministic_flags: ["subject_mismatch"],
      reasoning_categories: ["scope_mismatch"],
      suggested_fix: "Relink or revise the actor.",
      confidence_score: 0.2,
      created_at: "2026-04-06T11:58:00Z",
      support_snapshot_status: "versioned_v1",
      support_snapshot_note: null,
      support_snapshot_version: 1,
      support_snapshot: {
        claim_scope: {
          claim_id: claimId,
          draft_id: "draft-1",
          matter_id: "matter-1",
          evidence_bundle_id: "bundle-1",
          scope_kind: "bundle",
          allowed_source_document_ids: ["source-1", "source-2"],
        },
        valid_support_links: [
          {
            link_id: "link-1",
            claim_id: claimId,
            segment_id: "segment-1",
            source_document_id: "source-1",
            sequence_order: 1,
            link_type: "MANUAL",
            citation_text: null,
            user_confirmed: true,
            anchor: primaryAnchor,
            evidence_role: "primary",
          },
        ],
        excluded_support_links: [],
        support_items: [
          {
            order: 1,
            segment_id: "segment-1",
            source_document_id: "source-1",
            anchor: primaryAnchor,
            evidence_role: "primary",
            speaker: "A",
            segment_type: "ANSWER_BLOCK",
            raw_text: "A. Doe delivered the notice.",
            normalized_text: "a doe delivered the notice",
          },
        ],
        citations: [],
        provider_output: {
          primary_anchor: primaryAnchor,
          support_assessments: [
            {
              segment_id: "segment-1",
              anchor: primaryAnchor,
              role: "primary",
              contribution: "Direct support for the proposition.",
            },
          ],
        },
      },
    },
    previous_verification: {
      id: `verification-prev-${claimId}`,
      claim_unit_id: claimId,
      verdict: "ambiguous",
      reasoning: "Only contextual overlap was previously linked.",
      deterministic_flags: ["contextual_support_only"],
      reasoning_categories: ["weak_support"],
      suggested_fix: "Add direct testimony.",
      confidence_score: 0.4,
      created_at: "2026-04-05T11:58:00Z",
      support_snapshot_status: "versioned_v1",
      support_snapshot_note: null,
      support_snapshot_version: 1,
      support_snapshot: {
        claim_scope: {
          claim_id: claimId,
          draft_id: "draft-1",
          matter_id: "matter-1",
          evidence_bundle_id: "bundle-1",
          scope_kind: "bundle",
          allowed_source_document_ids: ["source-1", "source-2"],
        },
        valid_support_links: [],
        excluded_support_links: [],
        support_items: [],
        citations: [],
        provider_output: {
          primary_anchor: "p.10:1-10:2",
          support_assessments: [],
        },
      },
    },
    verification_runs: [],
    reasoning_categories: ["scope_mismatch"],
    contradiction_flags: [],
    claim_relationships: [],
    change_summary: {
      current_verdict: verdict,
      previous_verdict: "ambiguous",
      verdict_changed: true,
      current_confidence_score: 0.2,
      previous_confidence_score: 0.4,
      confidence_changed: true,
      current_primary_anchor: primaryAnchor,
      previous_primary_anchor: "p.10:1-10:2",
      primary_anchor_changed: true,
      support_changed: true,
      current_support_assessment_count: 1,
      previous_support_assessment_count: 0,
      current_excluded_link_count: 0,
      previous_excluded_link_count: 0,
      current_flags: ["subject_mismatch"],
      previous_flags: ["contextual_support_only"],
      flags_changed: true,
      current_reasoning_categories: ["scope_mismatch"],
      previous_reasoning_categories: ["weak_support"],
      reasoning_categories_changed: true,
      changed_since_last_run: true,
      latest_decision_at: null,
      latest_action: null,
    },
  }
}
