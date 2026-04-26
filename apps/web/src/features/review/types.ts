export type ReviewVerdict =
  | "supported"
  | "partially_supported"
  | "overstated"
  | "ambiguous"
  | "unsupported"
  | "unverified"

export type EvidenceRole = "primary" | "secondary" | "contextual"
export type ResolutionAction = "acknowledge_risk" | "mark_for_revision" | "resolve_with_edit"
export type QueueOrderMode = "severity" | "draft"
export type ReviewStateSource = "fresh_execution" | "persisted_read"
export type ReviewDisposition = "active" | "resolved"
export type DraftReviewRunStatus = "COMPLETED" | "FAILED"
export type StructuredReasoningCategory =
  | "temporal_mismatch"
  | "scope_mismatch"
  | "weak_support"
  | "contradiction"
  | "missing_authority"
  | "fabricated_authority"
export type ClaimRelationshipType = "supports" | "contradicts" | "depends_on" | "duplicate_of"

export interface DraftReviewSummary {
  draftId: string
  totalClaims: number
  totalFlaggedClaims: number
  remainingFlaggedClaims: number
  resolvedFlaggedClaims: number
  highestSeverity: ReviewVerdict | null
  topIssueCategories: string[]
}

export interface DraftReviewFreshness {
  stateSource: ReviewStateSource
  hasPersistedReviewRuns: boolean
  lastReviewRunAt: string | null
  latestReviewRunId: string | null
  latestReviewRunStatus: DraftReviewRunStatus | null
  latestDecisionAt: string | null
  hasDecisionsAfterLatestRun: boolean
  latestClaimVerificationAt: string | null
  latestVerificationRunId: string | null
  hasVerificationActivityAfterLatestRun: boolean
  isStale: boolean
}

export interface DraftReviewQueueState {
  draftId: string
  totalFlaggedClaims: number
  resolvedFlaggedClaims: number
  remainingFlaggedClaims: number
  nextClaimId: string | null
}

export interface DraftReviewRunSummary {
  runId: string
  status: DraftReviewRunStatus
  createdAt: string
  totalClaims: number
  totalFlaggedClaims: number
  resolvedFlaggedClaims: number
  remainingFlaggedClaims: number
  highestSeverityBucket: ReviewVerdict | null
}

export interface ClaimRelationshipSummary {
  relationshipType: ClaimRelationshipType
  relatedClaimId: string
  relatedClaimText: string
  reasonCode: string | null
  reasonText: string | null
  confidenceScore: number | null
}

export interface ClaimChangeSummary {
  currentVerdict: ReviewVerdict | null
  previousVerdict: ReviewVerdict | null
  verdictChanged: boolean
  currentConfidenceScore: number | null
  previousConfidenceScore: number | null
  confidenceChanged: boolean
  currentPrimaryAnchor: string | null
  previousPrimaryAnchor: string | null
  primaryAnchorChanged: boolean
  supportChanged: boolean
  currentSupportAssessmentCount: number
  previousSupportAssessmentCount: number
  currentExcludedLinkCount: number
  previousExcludedLinkCount: number
  currentFlags: string[]
  previousFlags: string[]
  flagsChanged: boolean
  currentReasoningCategories: StructuredReasoningCategory[]
  previousReasoningCategories: StructuredReasoningCategory[]
  reasoningCategoriesChanged: boolean
  changedSinceLastRun: boolean
}

export interface DraftReviewIntelligenceSummary {
  riskDistribution: {
    supported: number
    partially_supported: number
    overstated: number
    ambiguous: number
    unsupported: number
    unverified: number
  }
  mostUnstableClaimIds: string[]
  repeatedlyChangedClaimIds: string[]
  weakSupportClaimIds: string[]
  contradictionClaimIds: string[]
  contradictionPairCount: number
  duplicatePairCount: number
  weakSupportClusters: {
    flag: string
    claimCount: number
    claimIds: string[]
  }[]
}

export interface FlaggedClaimListItem {
  id: string
  draftSequence: number
  claimText: string
  verdict: ReviewVerdict
  deterministicFlags: string[]
  primaryAnchor: string | null
  confidenceScore: number | null
  supportCount: number
  latestVerificationRunAt: string | null
  changedSinceLastRun: boolean
  contradictionFlags: string[]
  reasoningCategories: StructuredReasoningCategory[]
}

export interface SupportAssessmentDetail {
  segmentId: string
  anchor: string | null
  role: EvidenceRole
  contribution: string
  snippet: string
}

export interface ExcludedLinkDetail {
  code: string | null
  reason: string | null
}

export interface EvidenceScopeDetail {
  scopeKind: string
  allowedSourceCount: number
}

export interface DecisionSummary {
  action: ResolutionAction
  note: string | null
  proposedReplacementText: string | null
  createdAt: string
}

export interface SelectedClaimDetail {
  id: string
  draftSequence: number
  claimText: string
  verdict: ReviewVerdict
  reviewDisposition: ReviewDisposition
  claimContext: string | null
  reasoning: string | null
  evidenceSummary: string
  suggestedFix: string | null
  deterministicFlags: string[]
  primaryAnchor: string | null
  confidenceScore: number | null
  reasoningCategories: StructuredReasoningCategory[]
  supportAssessments: SupportAssessmentDetail[]
  excludedLinks: ExcludedLinkDetail[]
  scope: EvidenceScopeDetail | null
  latestVerificationRunId: string | null
  latestVerificationRunAt: string | null
  latestDecision: DecisionSummary | null
  changedSinceLastRun: boolean
  changeSummary: ClaimChangeSummary | null
  contradictionFlags: string[]
  claimRelationships: ClaimRelationshipSummary[]
}

export interface ResolvedClaimListItem {
  id: string
  draftSequence: number
  claimText: string
  verdict: ReviewVerdict
  action: ResolutionAction
  note: string | null
  proposedReplacementText: string | null
  createdAt: string
}

export interface ResolutionActionState {
  selectedAction: ResolutionAction | null
  draftNote: string
  proposedClaimText: string
  dirty: boolean
  saving: boolean
}

export interface ResolutionSubmissionRequest {
  claimId: string
  action: ResolutionAction
  note: string | null
  proposedReplacementText: string | null
}

export interface VerificationSupportLinkHistory {
  linkId: string
  claimId: string
  segmentId: string
  sourceDocumentId: string | null
  sequenceOrder: number | null
  linkType: string
  citationText: string | null
  userConfirmed: boolean
  anchor: string | null
  evidenceRole: string | null
}

export interface ExcludedSupportLinkHistory {
  linkId: string
  claimId: string
  segmentId: string
  code: string | null
  message: string | null
}

export interface SupportItemHistory {
  order: number
  segmentId: string
  sourceDocumentId: string
  anchor: string
  evidenceRole: string
  speaker: string | null
  segmentType: string
  rawText: string
  normalizedText: string
}

export interface VerificationSupportSnapshotHistory {
  claimScope: {
    claimId: string
    draftId: string
    matterId: string
    evidenceBundleId: string | null
    scopeKind: string
    allowedSourceDocumentIds: string[]
  }
  validSupportLinks: VerificationSupportLinkHistory[]
  excludedSupportLinks: ExcludedSupportLinkHistory[]
  supportItems: SupportItemHistory[]
  citations: string[]
  providerOutput: {
    primaryAnchor: string | null
    supportAssessments: SupportAssessmentDetail[]
  }
}

export interface VerificationRunHistory {
  id: string
  claimUnitId: string
  verdict: ReviewVerdict
  reasoning: string
  deterministicFlags: string[]
  reasoningCategories: StructuredReasoningCategory[]
  suggestedFix: string | null
  confidenceScore: number | null
  createdAt: string
  supportSnapshotStatus: string | null
  supportSnapshotNote: string | null
  supportSnapshotVersion: number | null
  supportSnapshot: VerificationSupportSnapshotHistory | null
}

export interface ClaimReviewDecisionRecord extends DecisionSummary {
  id: string
  claimUnitId: string
  draftId: string
  verificationRunId: string | null
}

export interface ClaimReviewHistory {
  claimId: string
  draftId: string
  claimText: string
  assertionContext: string | null
  supportStatus: ReviewVerdict
  reviewDisposition: ReviewDisposition
  latestDecision: ClaimReviewDecisionRecord | null
  decisionHistory: ClaimReviewDecisionRecord[]
  latestVerification: VerificationRunHistory | null
  previousVerification: VerificationRunHistory | null
  verificationRuns: VerificationRunHistory[]
  reasoningCategories: StructuredReasoningCategory[]
  contradictionFlags: string[]
  claimRelationships: ClaimRelationshipSummary[]
  changeSummary: ClaimChangeSummary & {
    latestDecisionAt: string | null
    latestAction: ResolutionAction | null
  }
}

export interface ReviewWorkspaceData {
  summary: DraftReviewSummary
  freshness: DraftReviewFreshness
  queueState: DraftReviewQueueState
  latestReviewRun: DraftReviewRunSummary | null
  previousReviewRun: DraftReviewRunSummary | null
  intelligenceSummary: DraftReviewIntelligenceSummary | null
  activeClaims: FlaggedClaimListItem[]
  resolvedClaims: ResolvedClaimListItem[]
  detailsById: Record<string, SelectedClaimDetail>
}
