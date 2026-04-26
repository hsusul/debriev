import type {
  ClaimChangeSummary,
  ClaimReviewDecisionRecord,
  ClaimReviewHistory,
  ClaimRelationshipSummary,
  DraftReviewFreshness,
  DraftReviewIntelligenceSummary,
  DraftReviewQueueState,
  DraftReviewRunSummary,
  DraftReviewSummary,
  DecisionSummary,
  EvidenceRole,
  FlaggedClaimListItem,
  ResolutionActionState,
  ResolvedClaimListItem,
  ReviewDisposition,
  ReviewVerdict,
  ReviewWorkspaceData,
  SelectedClaimDetail,
  StructuredReasoningCategory,
  SupportAssessmentDetail,
  VerificationRunHistory,
  VerificationSupportSnapshotHistory,
} from "@/features/review/types"

export interface ReviewApiSupportAssessment {
  segment_id: string
  anchor: string
  role: EvidenceRole
  contribution: string
}

export interface ReviewApiExcludedLink {
  code: string | null
  message: string | null
}

export interface ReviewApiScope {
  scope_kind: string
  allowed_source_document_count: number
}

export interface ReviewApiFlaggedClaim {
  claim_id: string
  draft_sequence: number
  claim_text: string
  verdict: ReviewVerdict
  assertion_context: string | null
  reasoning: string | null
  deterministic_flags: string[]
  primary_anchor: string | null
  support_assessments: ReviewApiSupportAssessment[]
  excluded_links: ReviewApiExcludedLink[]
  scope: ReviewApiScope | null
  suggested_fix: string | null
  confidence_score: number | null
  latest_verification_run_id: string | null
  latest_verification_run_at: string | null
  reasoning_categories: StructuredReasoningCategory[]
  changed_since_last_run: boolean
  change_summary: ReviewApiClaimChangeSummary | null
  contradiction_flags: string[]
  claim_relationships: ReviewApiClaimRelationship[]
}

export interface ReviewApiResolvedClaim {
  claim: ReviewApiFlaggedClaim
  latest_decision: {
    action: ResolvedClaimListItem["action"]
    note: string | null
    proposed_replacement_text: string | null
    created_at: string
  }
}

export interface ReviewApiPayload {
  draft_id: string
  total_claims: number
  flagged_claim_counts: {
    total: number
  }
  review_overview: {
    highest_severity_bucket: ReviewVerdict | null
    top_issue_categories: string[]
  }
  freshness: {
    state_source: DraftReviewFreshness["stateSource"]
    has_persisted_review_runs: boolean
    last_review_run_at: string | null
    latest_review_run_id: string | null
    latest_review_run_status: DraftReviewFreshness["latestReviewRunStatus"]
    latest_decision_at: string | null
    has_decisions_after_latest_run: boolean
    latest_claim_verification_at: string | null
    latest_verification_run_id: string | null
    has_verification_activity_after_latest_run: boolean
    is_stale: boolean
  }
  queue_state: {
    draft_id: string
    total_flagged_claims: number
    resolved_flagged_claims: number
    remaining_flagged_claims: number
    next_claim_id: string | null
  }
  active_queue_claims: ReviewApiFlaggedClaim[]
  resolved_claims: ReviewApiResolvedClaim[]
  latest_review_run: ReviewApiRunSummary | null
  previous_review_run: ReviewApiRunSummary | null
  intelligence_summary: ReviewApiIntelligenceSummary | null
}

interface ReviewApiClaimRelationship {
  relationship_type: ClaimRelationshipSummary["relationshipType"]
  related_claim_id: string
  related_claim_text: string
  reason_code: string | null
  reason_text: string | null
  confidence_score: number | null
}

interface ReviewApiClaimChangeSummary {
  current_verdict: ReviewVerdict | null
  previous_verdict: ReviewVerdict | null
  verdict_changed: boolean
  current_confidence_score: number | null
  previous_confidence_score: number | null
  confidence_changed: boolean
  current_primary_anchor: string | null
  previous_primary_anchor: string | null
  primary_anchor_changed: boolean
  support_changed: boolean
  current_support_assessment_count: number
  previous_support_assessment_count: number
  current_excluded_link_count: number
  previous_excluded_link_count: number
  current_flags: string[]
  previous_flags: string[]
  flags_changed: boolean
  current_reasoning_categories: StructuredReasoningCategory[]
  previous_reasoning_categories: StructuredReasoningCategory[]
  reasoning_categories_changed: boolean
  changed_since_last_run: boolean
}

interface ReviewApiIntelligenceSummary {
  risk_distribution: DraftReviewIntelligenceSummary["riskDistribution"]
  most_unstable_claim_ids: string[]
  repeatedly_changed_claim_ids: string[]
  weak_support_claim_ids: string[]
  contradiction_claim_ids: string[]
  contradiction_pair_count: number
  duplicate_pair_count: number
  weak_support_clusters: {
    flag: string
    claim_count: number
    claim_ids: string[]
  }[]
}

export interface ReviewApiRunSummary {
  run_id: string
  status: DraftReviewRunSummary["status"]
  created_at: string
  total_claims: number
  total_flagged_claims: number
  resolved_flagged_claims: number
  remaining_flagged_claims: number
  highest_severity_bucket: ReviewVerdict | null
}

export interface ClaimReviewHistoryApiPayload {
  claim_id: string
  draft_id: string
  claim_text: string
  assertion_context: string | null
  support_status: ReviewVerdict
  review_disposition: ReviewDisposition
  latest_decision: ReviewApiDecision | null
  decision_history: ReviewApiDecision[]
  latest_verification: ReviewApiVerificationRun | null
  previous_verification: ReviewApiVerificationRun | null
  verification_runs: ReviewApiVerificationRun[]
  reasoning_categories: StructuredReasoningCategory[]
  contradiction_flags: string[]
  claim_relationships: ReviewApiClaimRelationship[]
  change_summary: ReviewApiClaimChangeSummary & {
    latest_decision_at: string | null
    latest_action: DecisionSummary["action"] | null
  }
}

interface ReviewApiDecision {
  id: string
  claim_unit_id: string
  draft_id: string
  verification_run_id: string | null
  action: DecisionSummary["action"]
  note: string | null
  proposed_replacement_text: string | null
  created_at: string
}

interface ReviewApiVerificationRun {
  id: string
  claim_unit_id: string
  verdict: ReviewVerdict
  reasoning: string
  deterministic_flags: string[]
  reasoning_categories: StructuredReasoningCategory[]
  suggested_fix: string | null
  confidence_score: number | null
  created_at: string
  support_snapshot_status: string | null
  support_snapshot_note: string | null
  support_snapshot_version: number | null
  support_snapshot: ReviewApiVerificationSupportSnapshot | null
}

interface ReviewApiVerificationSupportSnapshot {
  claim_scope: {
    claim_id: string
    draft_id: string
    matter_id: string
    evidence_bundle_id: string | null
    scope_kind: string
    allowed_source_document_ids: string[]
  }
  valid_support_links: {
    link_id: string
    claim_id: string
    segment_id: string
    source_document_id: string | null
    sequence_order: number | null
    link_type: string
    citation_text: string | null
    user_confirmed: boolean
    anchor: string | null
    evidence_role: string | null
  }[]
  excluded_support_links: {
    link_id: string
    claim_id: string
    segment_id: string
    code: string | null
    message: string | null
  }[]
  support_items: {
    order: number
    segment_id: string
    source_document_id: string
    anchor: string
    evidence_role: string
    speaker: string | null
    segment_type: string
    raw_text: string
    normalized_text: string
  }[]
  citations: string[]
  provider_output: {
    primary_anchor: string | null
    support_assessments: ReviewApiSupportAssessment[]
  }
}

export function buildWorkspaceData(payload: ReviewApiPayload): ReviewWorkspaceData {
  const activeClaims = payload.active_queue_claims.map(adaptFlaggedClaim)
  const resolvedClaims = payload.resolved_claims.map(adaptResolvedClaim)
  const allClaimDetails = [
    ...payload.active_queue_claims.map((item) => [item.claim_id, adaptSelectedClaimDetail(item, "active", null)]),
    ...payload.resolved_claims.map((item) => [
      item.claim.claim_id,
      adaptSelectedClaimDetail(item.claim, "resolved", adaptDecisionSummary(item.latest_decision)),
    ]),
  ]
  const detailsById = Object.fromEntries(
    allClaimDetails,
  )

  return {
    summary: adaptReviewSummary(payload),
    freshness: adaptFreshness(payload.freshness),
    queueState: adaptQueueState(payload.queue_state),
    latestReviewRun: adaptRunSummary(payload.latest_review_run),
    previousReviewRun: adaptRunSummary(payload.previous_review_run),
    intelligenceSummary: adaptIntelligenceSummary(payload.intelligence_summary),
    activeClaims,
    resolvedClaims,
    detailsById,
  }
}

export function adaptClaimReviewHistory(payload: ClaimReviewHistoryApiPayload): ClaimReviewHistory {
  const changeSummary = adaptClaimChangeSummary(payload.change_summary)
  return {
    claimId: payload.claim_id,
    draftId: payload.draft_id,
    claimText: payload.claim_text,
    assertionContext: payload.assertion_context,
    supportStatus: payload.support_status,
    reviewDisposition: payload.review_disposition,
    latestDecision: adaptDecisionRecord(payload.latest_decision),
    decisionHistory: payload.decision_history.map(adaptDecisionRecord).filter(Boolean) as ClaimReviewDecisionRecord[],
    latestVerification: adaptVerificationRun(payload.latest_verification),
    previousVerification: adaptVerificationRun(payload.previous_verification),
    verificationRuns: payload.verification_runs.map(adaptVerificationRun).filter(Boolean) as VerificationRunHistory[],
    reasoningCategories: payload.reasoning_categories,
    contradictionFlags: payload.contradiction_flags,
    claimRelationships: payload.claim_relationships.map(adaptClaimRelationship),
    changeSummary: {
      ...(changeSummary ?? {
        currentVerdict: null,
        previousVerdict: null,
        verdictChanged: false,
        currentConfidenceScore: null,
        previousConfidenceScore: null,
        confidenceChanged: false,
        currentPrimaryAnchor: null,
        previousPrimaryAnchor: null,
        primaryAnchorChanged: false,
        supportChanged: false,
        currentSupportAssessmentCount: 0,
        previousSupportAssessmentCount: 0,
        currentExcludedLinkCount: 0,
        previousExcludedLinkCount: 0,
        currentFlags: [],
        previousFlags: [],
        flagsChanged: false,
        currentReasoningCategories: [],
        previousReasoningCategories: [],
        reasoningCategoriesChanged: false,
        changedSinceLastRun: false,
      }),
      latestDecisionAt: payload.change_summary.latest_decision_at,
      latestAction: payload.change_summary.latest_action,
    },
  }
}

export function buildDefaultResolutionState(item: SelectedClaimDetail): ResolutionActionState {
  switch (item.verdict) {
    case "overstated":
    case "partially_supported":
      return {
        selectedAction: "resolve_with_edit",
        draftNote: item.suggestedFix ?? "",
        proposedClaimText: item.claimText,
        dirty: false,
        saving: false,
      }
    case "unsupported":
    case "ambiguous":
      return {
        selectedAction: "mark_for_revision",
        draftNote: item.suggestedFix ?? "",
        proposedClaimText: item.claimText,
        dirty: false,
        saving: false,
      }
    case "supported":
    case "unverified":
      return {
        selectedAction: "acknowledge_risk",
        draftNote: item.suggestedFix ?? "",
        proposedClaimText: item.claimText,
        dirty: false,
        saving: false,
      }
  }
}

function adaptReviewSummary(payload: ReviewApiPayload): DraftReviewSummary {
  return {
    draftId: payload.draft_id,
    totalClaims: payload.total_claims,
    totalFlaggedClaims: payload.flagged_claim_counts.total,
    remainingFlaggedClaims: payload.queue_state.remaining_flagged_claims,
    resolvedFlaggedClaims: payload.queue_state.resolved_flagged_claims,
    highestSeverity: payload.review_overview.highest_severity_bucket,
    topIssueCategories: payload.review_overview.top_issue_categories,
  }
}

function adaptFreshness(payload: ReviewApiPayload["freshness"]): DraftReviewFreshness {
  return {
    stateSource: payload.state_source,
    hasPersistedReviewRuns: payload.has_persisted_review_runs,
    lastReviewRunAt: payload.last_review_run_at,
    latestReviewRunId: payload.latest_review_run_id,
    latestReviewRunStatus: payload.latest_review_run_status,
    latestDecisionAt: payload.latest_decision_at,
    hasDecisionsAfterLatestRun: payload.has_decisions_after_latest_run,
    latestClaimVerificationAt: payload.latest_claim_verification_at,
    latestVerificationRunId: payload.latest_verification_run_id,
    hasVerificationActivityAfterLatestRun: payload.has_verification_activity_after_latest_run,
    isStale: payload.is_stale,
  }
}

function adaptRunSummary(payload: ReviewApiRunSummary | null): DraftReviewRunSummary | null {
  if (payload == null) {
    return null
  }
  return {
    runId: payload.run_id,
    status: payload.status,
    createdAt: payload.created_at,
    totalClaims: payload.total_claims,
    totalFlaggedClaims: payload.total_flagged_claims,
    resolvedFlaggedClaims: payload.resolved_flagged_claims,
    remainingFlaggedClaims: payload.remaining_flagged_claims,
    highestSeverityBucket: payload.highest_severity_bucket,
  }
}

function adaptQueueState(payload: ReviewApiPayload["queue_state"]): DraftReviewQueueState {
  return {
    draftId: payload.draft_id,
    totalFlaggedClaims: payload.total_flagged_claims,
    resolvedFlaggedClaims: payload.resolved_flagged_claims,
    remainingFlaggedClaims: payload.remaining_flagged_claims,
    nextClaimId: payload.next_claim_id,
  }
}

function adaptIntelligenceSummary(
  payload: ReviewApiIntelligenceSummary | null,
): DraftReviewIntelligenceSummary | null {
  if (payload == null) {
    return null
  }
  return {
    riskDistribution: payload.risk_distribution,
    mostUnstableClaimIds: payload.most_unstable_claim_ids,
    repeatedlyChangedClaimIds: payload.repeatedly_changed_claim_ids,
    weakSupportClaimIds: payload.weak_support_claim_ids,
    contradictionClaimIds: payload.contradiction_claim_ids,
    contradictionPairCount: payload.contradiction_pair_count,
    duplicatePairCount: payload.duplicate_pair_count,
    weakSupportClusters: payload.weak_support_clusters.map((cluster) => ({
      flag: cluster.flag,
      claimCount: cluster.claim_count,
      claimIds: cluster.claim_ids,
    })),
  }
}

function adaptFlaggedClaim(item: ReviewApiFlaggedClaim): FlaggedClaimListItem {
  return {
    id: item.claim_id,
    draftSequence: item.draft_sequence,
    claimText: item.claim_text,
    verdict: item.verdict,
    deterministicFlags: item.deterministic_flags,
    primaryAnchor: item.primary_anchor,
    confidenceScore: item.confidence_score,
    supportCount: item.support_assessments.length,
    latestVerificationRunAt: item.latest_verification_run_at,
    changedSinceLastRun: item.changed_since_last_run,
    contradictionFlags: item.contradiction_flags,
    reasoningCategories: item.reasoning_categories,
  }
}

function adaptResolvedClaim(item: ReviewApiResolvedClaim): ResolvedClaimListItem {
  return {
    id: item.claim.claim_id,
    draftSequence: item.claim.draft_sequence,
    claimText: item.claim.claim_text,
    verdict: item.claim.verdict,
    action: item.latest_decision.action,
    note: item.latest_decision.note,
    proposedReplacementText: item.latest_decision.proposed_replacement_text,
    createdAt: item.latest_decision.created_at,
  }
}

function adaptSelectedClaimDetail(
  item: ReviewApiFlaggedClaim,
  reviewDisposition: ReviewDisposition,
  latestDecision: DecisionSummary | null,
): SelectedClaimDetail {
  const supportAssessments: SupportAssessmentDetail[] = item.support_assessments.map((assessment) => ({
    segmentId: assessment.segment_id,
    anchor: assessment.anchor,
    role: assessment.role,
    contribution: assessment.contribution,
    snippet: assessment.contribution,
  }))

  return {
    id: item.claim_id,
    draftSequence: item.draft_sequence,
    claimText: item.claim_text,
    verdict: item.verdict,
    reviewDisposition,
    claimContext: item.assertion_context,
    reasoning: item.reasoning,
    evidenceSummary: buildEvidenceSummary(item),
    suggestedFix: item.suggested_fix,
    deterministicFlags: item.deterministic_flags,
    primaryAnchor: item.primary_anchor,
    confidenceScore: item.confidence_score,
    reasoningCategories: item.reasoning_categories,
    supportAssessments,
    excludedLinks: item.excluded_links.map((link) => ({
      code: link.code,
      reason: link.message,
    })),
    scope: item.scope
      ? {
          scopeKind: item.scope.scope_kind,
          allowedSourceCount: item.scope.allowed_source_document_count,
        }
      : null,
    latestVerificationRunId: item.latest_verification_run_id,
    latestVerificationRunAt: item.latest_verification_run_at,
    latestDecision,
    changedSinceLastRun: item.changed_since_last_run,
    changeSummary: adaptClaimChangeSummary(item.change_summary),
    contradictionFlags: item.contradiction_flags,
    claimRelationships: item.claim_relationships.map(adaptClaimRelationship),
  }
}

function adaptDecisionSummary(payload: ReviewApiResolvedClaim["latest_decision"]): DecisionSummary {
  return {
    action: payload.action,
    note: payload.note,
    proposedReplacementText: payload.proposed_replacement_text,
    createdAt: payload.created_at,
  }
}

function adaptDecisionRecord(payload: ReviewApiDecision | null): ClaimReviewDecisionRecord | null {
  if (payload == null) {
    return null
  }
  return {
    id: payload.id,
    claimUnitId: payload.claim_unit_id,
    draftId: payload.draft_id,
    verificationRunId: payload.verification_run_id,
    action: payload.action,
    note: payload.note,
    proposedReplacementText: payload.proposed_replacement_text,
    createdAt: payload.created_at,
  }
}

function adaptVerificationRun(payload: ReviewApiVerificationRun | null): VerificationRunHistory | null {
  if (payload == null) {
    return null
  }
  return {
    id: payload.id,
    claimUnitId: payload.claim_unit_id,
    verdict: payload.verdict,
    reasoning: payload.reasoning,
    deterministicFlags: payload.deterministic_flags,
    reasoningCategories: payload.reasoning_categories,
    suggestedFix: payload.suggested_fix,
    confidenceScore: payload.confidence_score,
    createdAt: payload.created_at,
    supportSnapshotStatus: payload.support_snapshot_status,
    supportSnapshotNote: payload.support_snapshot_note,
    supportSnapshotVersion: payload.support_snapshot_version,
    supportSnapshot: adaptVerificationSupportSnapshot(payload.support_snapshot),
  }
}

function adaptClaimRelationship(payload: ReviewApiClaimRelationship): ClaimRelationshipSummary {
  return {
    relationshipType: payload.relationship_type,
    relatedClaimId: payload.related_claim_id,
    relatedClaimText: payload.related_claim_text,
    reasonCode: payload.reason_code,
    reasonText: payload.reason_text,
    confidenceScore: payload.confidence_score,
  }
}

function adaptClaimChangeSummary(payload: ReviewApiClaimChangeSummary | null): ClaimChangeSummary | null {
  if (payload == null) {
    return null
  }
  return {
    currentVerdict: payload.current_verdict,
    previousVerdict: payload.previous_verdict,
    verdictChanged: payload.verdict_changed,
    currentConfidenceScore: payload.current_confidence_score,
    previousConfidenceScore: payload.previous_confidence_score,
    confidenceChanged: payload.confidence_changed,
    currentPrimaryAnchor: payload.current_primary_anchor,
    previousPrimaryAnchor: payload.previous_primary_anchor,
    primaryAnchorChanged: payload.primary_anchor_changed,
    supportChanged: payload.support_changed,
    currentSupportAssessmentCount: payload.current_support_assessment_count,
    previousSupportAssessmentCount: payload.previous_support_assessment_count,
    currentExcludedLinkCount: payload.current_excluded_link_count,
    previousExcludedLinkCount: payload.previous_excluded_link_count,
    currentFlags: payload.current_flags,
    previousFlags: payload.previous_flags,
    flagsChanged: payload.flags_changed,
    currentReasoningCategories: payload.current_reasoning_categories,
    previousReasoningCategories: payload.previous_reasoning_categories,
    reasoningCategoriesChanged: payload.reasoning_categories_changed,
    changedSinceLastRun: payload.changed_since_last_run,
  }
}

function adaptVerificationSupportSnapshot(
  payload: ReviewApiVerificationSupportSnapshot | null,
): VerificationSupportSnapshotHistory | null {
  if (payload == null) {
    return null
  }

  return {
    claimScope: {
      claimId: payload.claim_scope.claim_id,
      draftId: payload.claim_scope.draft_id,
      matterId: payload.claim_scope.matter_id,
      evidenceBundleId: payload.claim_scope.evidence_bundle_id,
      scopeKind: payload.claim_scope.scope_kind,
      allowedSourceDocumentIds: payload.claim_scope.allowed_source_document_ids,
    },
    validSupportLinks: payload.valid_support_links.map((link) => ({
      linkId: link.link_id,
      claimId: link.claim_id,
      segmentId: link.segment_id,
      sourceDocumentId: link.source_document_id,
      sequenceOrder: link.sequence_order,
      linkType: link.link_type,
      citationText: link.citation_text,
      userConfirmed: link.user_confirmed,
      anchor: link.anchor,
      evidenceRole: link.evidence_role,
    })),
    excludedSupportLinks: payload.excluded_support_links.map((link) => ({
      linkId: link.link_id,
      claimId: link.claim_id,
      segmentId: link.segment_id,
      code: link.code,
      message: link.message,
    })),
    supportItems: payload.support_items.map((item) => ({
      order: item.order,
      segmentId: item.segment_id,
      sourceDocumentId: item.source_document_id,
      anchor: item.anchor,
      evidenceRole: item.evidence_role,
      speaker: item.speaker,
      segmentType: item.segment_type,
      rawText: item.raw_text,
      normalizedText: item.normalized_text,
    })),
    citations: [...payload.citations],
    providerOutput: {
      primaryAnchor: payload.provider_output.primary_anchor,
      supportAssessments: payload.provider_output.support_assessments.map((assessment) => ({
        segmentId: assessment.segment_id,
        anchor: assessment.anchor,
        role: assessment.role,
        contribution: assessment.contribution,
        snippet: assessment.contribution,
      })),
    },
  }
}

function buildEvidenceSummary(item: ReviewApiFlaggedClaim) {
  const parts: string[] = []
  if (item.primary_anchor) {
    parts.push(`Primary anchor ${item.primary_anchor}.`)
  }
  if (item.support_assessments.length > 0) {
    parts.push(`${item.support_assessments.length} support assessment${pluralize(item.support_assessments.length)}.`)
  }
  if (item.excluded_links.length > 0) {
    parts.push(`${item.excluded_links.length} excluded link${pluralize(item.excluded_links.length)}.`)
  }
  if (item.scope) {
    parts.push(`${formatScopeKind(item.scope.scope_kind)} scope.`)
  }
  if (parts.length === 0 && item.reasoning) {
    return item.reasoning
  }
  return parts.join(" ")
}

function formatScopeKind(scopeKind: string) {
  return scopeKind.replace(/_/g, " ")
}

function pluralize(count: number) {
  return count === 1 ? "" : "s"
}
