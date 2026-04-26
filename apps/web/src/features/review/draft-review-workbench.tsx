import { useEffect, useMemo, useState } from "react"
import { AlertCircle, LoaderCircle } from "lucide-react"

import { AppLayout } from "@/components/layout/app-layout"
import { WorkspaceLayout } from "@/components/layout/workspace-layout"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { adaptClaimReviewHistory, buildDefaultResolutionState, buildWorkspaceData } from "@/features/review/adapters"
import {
  fetchClaimReviewHistory,
  fetchDraftReviewState,
  rerunDraftReview,
  resolveDraftId,
  submitClaimDecision,
} from "@/features/review/api"
import type {
  ClaimReviewHistory,
  DraftReviewFreshness,
  DraftReviewRunSummary,
  FlaggedClaimListItem,
  QueueOrderMode,
  ResolutionAction,
  ResolutionActionState,
  ResolutionSubmissionRequest,
  ResolvedClaimListItem,
  ReviewWorkspaceData,
  ReviewVerdict,
  SelectedClaimDetail,
} from "@/features/review/types"
import { EvidenceInspector } from "./workspace/evidence-inspector"
import { DocumentReader } from "./workspace/document-reader"
import { ReviewActionBar } from "./workspace/review-action-bar"
import { ReviewQueue, type QueueSection } from "./workspace/review-queue"

const verdictOrder: ReviewVerdict[] = [
  "unsupported",
  "overstated",
  "ambiguous",
  "unverified",
  "partially_supported",
  "supported",
]

export function DraftReviewWorkbench() {
  const draftId = useMemo(() => resolveDraftId(), [])
  const [workspaceData, setWorkspaceData] = useState<ReviewWorkspaceData | null>(null)
  const [selectedClaimId, setSelectedClaimId] = useState("")
  const [verdictFilter, setVerdictFilter] = useState<ReviewVerdict | "all">("all")
  const [queueOrderMode, setQueueOrderMode] = useState<QueueOrderMode>("severity")
  const [resolutionStateById, setResolutionStateById] = useState<Record<string, ResolutionActionState>>({})
  const [claimHistoryById, setClaimHistoryById] = useState<Record<string, ClaimReviewHistory>>({})
  const [claimHistoryErrorById, setClaimHistoryErrorById] = useState<Record<string, string>>({})
  const [loadingClaimHistoryId, setLoadingClaimHistoryId] = useState<string | null>(null)
  const [isHydrating, setIsHydrating] = useState(draftId != null)
  const [isRerunning, setIsRerunning] = useState(false)
  const [backendError, setBackendError] = useState<string | null>(null)

  useEffect(() => {
    const resolvedDraftId = draftId
    if (resolvedDraftId == null) {
      setIsHydrating(false)
      return
    }

    const requestedDraftId: string = resolvedDraftId
    let active = true

    async function loadInitialReviewState() {
      setIsHydrating(true)
      setBackendError(null)

      try {
        const payload = await fetchDraftReviewState(requestedDraftId)
        if (!active) {
          return
        }
        applyWorkspacePayload(payload, payload.queue_state.next_claim_id)
      } catch (error) {
        if (!active) {
          return
        }
        setBackendError(getErrorMessage(error))
      } finally {
        if (active) {
          setIsHydrating(false)
        }
      }
    }

    void loadInitialReviewState()

    return () => {
      active = false
    }
  }, [draftId])

  const activeClaims = workspaceData?.activeClaims ?? []
  const resolvedClaims = workspaceData?.resolvedClaims ?? []

  const filteredClaims = useMemo(() => {
    if (verdictFilter === "all") {
      return activeClaims
    }

    return activeClaims.filter((claim) => claim.verdict === verdictFilter)
  }, [activeClaims, verdictFilter])

  const orderedVisibleClaims = useMemo(
    () => orderClaims(filteredClaims, queueOrderMode),
    [filteredClaims, queueOrderMode],
  )

  const queueSections = useMemo(
    () => buildQueueSections(orderedVisibleClaims, queueOrderMode),
    [orderedVisibleClaims, queueOrderMode],
  )

  const verdictCounts = useMemo(
    () =>
      activeClaims.reduce(
        (counts, claim) => {
          counts[claim.verdict] += 1
          return counts
        },
        {
          supported: 0,
          partially_supported: 0,
          overstated: 0,
          ambiguous: 0,
          unsupported: 0,
          unverified: 0,
        } satisfies Record<ReviewVerdict, number>,
      ),
    [activeClaims],
  )

  useEffect(() => {
    const selectableClaimIds = [
      ...orderedVisibleClaims.map((claim) => claim.id),
      ...resolvedClaims.map((claim) => claim.id),
    ]

    if (selectableClaimIds.length === 0) {
      if (selectedClaimId !== "") {
        setSelectedClaimId("")
      }
      return
    }

    if (!selectableClaimIds.includes(selectedClaimId)) {
      setSelectedClaimId(orderedVisibleClaims[0]?.id ?? resolvedClaims[0]?.id ?? "")
    }
  }, [orderedVisibleClaims, resolvedClaims, selectedClaimId])

  const selectedQueueItem =
    orderedVisibleClaims.find((claim) => claim.id === selectedClaimId) ?? orderedVisibleClaims[0] ?? null
  const selectedResolvedClaim = resolvedClaims.find((claim) => claim.id === selectedClaimId) ?? null
  const selectedClaim =
    workspaceData != null && selectedClaimId ? workspaceData.detailsById[selectedClaimId] ?? null : null
  const selectedClaimHistory = selectedClaim != null ? claimHistoryById[selectedClaim.id] ?? null : null
  const selectedClaimHistoryError = selectedClaim != null ? claimHistoryErrorById[selectedClaim.id] ?? null : null
  const isClaimHistoryLoading = selectedClaim != null && loadingClaimHistoryId === selectedClaim.id
  const selectedIndex =
    selectedQueueItem != null ? orderedVisibleClaims.findIndex((claim) => claim.id === selectedQueueItem.id) : -1
  const hasPrevious = selectedIndex > 0
  const hasNext = selectedIndex >= 0 && selectedIndex < orderedVisibleClaims.length - 1

  const selectedResolution =
    selectedClaim != null && selectedClaim.reviewDisposition === "active"
      ? resolutionStateById[selectedClaim.id] ?? buildDefaultResolutionState(selectedClaim)
      : null

  useEffect(() => {
    function handleKeydown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null
      const tagName = target?.tagName?.toLowerCase()
      const editing =
        tagName === "input" || tagName === "textarea" || target?.isContentEditable === true

      if (editing) {
        return
      }

      if (event.key === "j" && hasNext) {
        event.preventDefault()
        moveSelection(1)
      }

      if (event.key === "k" && hasPrevious) {
        event.preventDefault()
        moveSelection(-1)
      }
    }

    window.addEventListener("keydown", handleKeydown)
    return () => window.removeEventListener("keydown", handleKeydown)
  }, [hasNext, hasPrevious, selectedIndex, orderedVisibleClaims])

  const needsFreshReview = workspaceData != null && !workspaceData.freshness.hasPersistedReviewRuns

  useEffect(() => {
    if (selectedClaim == null || claimHistoryById[selectedClaim.id] != null || isClaimHistoryLoading) {
      return
    }

    const targetClaimId = selectedClaim.id
    let active = true
    setLoadingClaimHistoryId(targetClaimId)
    setClaimHistoryErrorById((current) => {
      const next = { ...current }
      delete next[targetClaimId]
      return next
    })

    async function loadClaimHistory() {
      try {
        const payload = await fetchClaimReviewHistory(targetClaimId)
        if (!active) {
          return
        }
        setClaimHistoryById((current) => ({
          ...current,
          [targetClaimId]: adaptClaimReviewHistory(payload),
        }))
      } catch (error) {
        if (!active) {
          return
        }
        setClaimHistoryErrorById((current) => ({
          ...current,
          [targetClaimId]: getErrorMessage(error),
        }))
      } finally {
        if (active) {
          setLoadingClaimHistoryId((current) => (current === targetClaimId ? null : current))
        }
      }
    }

    void loadClaimHistory()

    return () => {
      active = false
    }
  }, [claimHistoryById, isClaimHistoryLoading, selectedClaim])

  function applyWorkspacePayload(payload: Parameters<typeof buildWorkspaceData>[0], preferredClaimId?: string | null) {
    const nextWorkspaceData = buildWorkspaceData(payload)

    setWorkspaceData(nextWorkspaceData)
    setClaimHistoryById({})
    setClaimHistoryErrorById({})
    setLoadingClaimHistoryId(null)
    setResolutionStateById((current) => mergeResolutionStates(current, nextWorkspaceData))
    setSelectedClaimId((current) =>
      resolveNextSelectedClaimId({
        currentSelectedClaimId: current,
        preferredClaimId,
        workspaceData: nextWorkspaceData,
      }),
    )
  }

  function updateResolutionAction(nextAction: ResolutionAction) {
    if (selectedClaim == null || selectedClaim.reviewDisposition !== "active" || selectedResolution == null) {
      return
    }

    setResolutionStateById((current) => ({
      ...current,
      [selectedClaim.id]: {
        ...selectedResolution,
        selectedAction: nextAction,
        dirty: true,
      },
    }))
  }

  function updateResolutionNote(nextNote: string) {
    if (selectedClaim == null || selectedClaim.reviewDisposition !== "active" || selectedResolution == null) {
      return
    }

    setResolutionStateById((current) => ({
      ...current,
      [selectedClaim.id]: {
        ...selectedResolution,
        draftNote: nextNote,
        dirty: true,
      },
    }))
  }

  function updateProposedClaimText(nextText: string) {
    if (selectedClaim == null || selectedClaim.reviewDisposition !== "active" || selectedResolution == null) {
      return
    }

    setResolutionStateById((current) => ({
      ...current,
      [selectedClaim.id]: {
        ...selectedResolution,
        proposedClaimText: nextText,
        dirty: true,
      },
    }))
  }

  function moveSelection(step: -1 | 1) {
    const nextIndex = selectedIndex + step
    const nextClaim = orderedVisibleClaims[nextIndex]
    if (nextClaim != null) {
      setSelectedClaimId(nextClaim.id)
    }
  }

  async function hydrateReviewState(preferredClaimId?: string | null) {
    if (draftId == null) {
      return
    }

    setBackendError(null)
    setIsHydrating(workspaceData == null)

    try {
      const payload = await fetchDraftReviewState(draftId)
      applyWorkspacePayload(payload, preferredClaimId ?? payload.queue_state.next_claim_id)
    } catch (error) {
      setBackendError(getErrorMessage(error))
    } finally {
      setIsHydrating(false)
    }
  }

  async function handleRerunReview() {
    if (draftId == null) {
      return
    }

    setBackendError(null)
    setIsRerunning(true)

    try {
      const payload = await rerunDraftReview(draftId)
      applyWorkspacePayload(payload, payload.queue_state.next_claim_id)
    } catch (error) {
      setBackendError(getErrorMessage(error))
    } finally {
      setIsRerunning(false)
    }
  }

  async function submitResolution() {
    if (selectedClaim == null || selectedClaim.reviewDisposition !== "active" || selectedResolution == null) {
      return
    }

    const submission = buildResolutionSubmission(selectedClaim, selectedResolution)
    if (submission == null) {
      return
    }

    setBackendError(null)
    setResolutionStateById((current) => ({
      ...current,
      [selectedClaim.id]: {
        ...selectedResolution,
        saving: true,
      },
    }))

    try {
      const mutation = await submitClaimDecision(submission)
      await hydrateReviewState(mutation.draft_queue.next_claim_id)
    } catch (error) {
      setBackendError(getErrorMessage(error))
      setResolutionStateById((current) => ({
        ...current,
        [selectedClaim.id]: {
          ...selectedResolution,
          saving: false,
        },
      }))
    }
  }

  return (
    <AppLayout>
      <WorkspaceLayout
        leftPane={
          <ReviewQueue
            queueSections={queueSections}
            resolutionStateById={resolutionStateById}
            selectedClaimId={selectedClaim?.id ?? ""}
            selectedDisposition={selectedClaim?.reviewDisposition ?? null}
            selectedIndex={selectedIndex}
            visibleClaimCount={orderedVisibleClaims.length}
            verdictFilter={verdictFilter}
            queueOrderMode={queueOrderMode}
            verdictCounts={verdictCounts}
            resolvedClaims={resolvedClaims}
            freshness={workspaceData?.freshness ?? null}
            latestReviewRun={workspaceData?.latestReviewRun ?? null}
            previousReviewRun={workspaceData?.previousReviewRun ?? null}
            intelligenceSummary={workspaceData?.intelligenceSummary ?? null}
            isHydrating={isHydrating}
            isRerunning={isRerunning}
            onSelectClaim={setSelectedClaimId}
            onFilterChange={setVerdictFilter}
            onQueueOrderModeChange={setQueueOrderMode}
            onRerun={handleRerunReview}
          />
        }
        centerPane={
          <ClaimFocusPane
            draftId={draftId}
            claim={selectedClaim}
            claimHistory={selectedClaimHistory}
            resolvedClaim={selectedResolvedClaim}
            selectedIndex={selectedIndex}
            visibleClaimCount={orderedVisibleClaims.length}
            nextClaim={orderedVisibleClaims[selectedIndex + 1] ?? null}
            resolutionState={selectedResolution}
            hasPrevious={hasPrevious}
            hasNext={hasNext}
            isHydrating={isHydrating}
            isRerunning={isRerunning}
            isClaimHistoryLoading={isClaimHistoryLoading}
            backendError={backendError}
            claimHistoryError={selectedClaimHistoryError}
            needsFreshReview={needsFreshReview}
            freshness={workspaceData?.freshness ?? null}
            latestReviewRun={workspaceData?.latestReviewRun ?? null}
            onPrevious={() => moveSelection(-1)}
            onNext={() => moveSelection(1)}
            onResolutionAction={updateResolutionAction}
            onResolutionNoteChange={updateResolutionNote}
            onProposedClaimTextChange={updateProposedClaimText}
            onSubmitResolution={submitResolution}
            onRetry={hydrateReviewState}
            onRerun={handleRerunReview}
          />
        }
        rightPane={
          <EvidenceInspector
            claim={selectedClaim}
            claimHistory={selectedClaimHistory}
            isLoading={isClaimHistoryLoading}
          />
        }
      />
    </AppLayout>
  )
}

function ClaimFocusPane({
  draftId,
  claim,
  claimHistory,
  resolvedClaim,
  selectedIndex,
  visibleClaimCount,
  nextClaim,
  resolutionState,
  hasPrevious,
  hasNext,
  isHydrating,
  isRerunning,
  isClaimHistoryLoading,
  backendError,
  claimHistoryError,
  needsFreshReview,
  freshness,
  latestReviewRun,
  onPrevious,
  onNext,
  onResolutionAction,
  onResolutionNoteChange,
  onProposedClaimTextChange,
  onSubmitResolution,
  onRetry,
  onRerun,
}: {
  draftId: string | null
  claim: SelectedClaimDetail | null
  claimHistory: ClaimReviewHistory | null
  resolvedClaim: ResolvedClaimListItem | null
  selectedIndex: number
  visibleClaimCount: number
  nextClaim: FlaggedClaimListItem | null
  resolutionState: ResolutionActionState | null
  hasPrevious: boolean
  hasNext: boolean
  isHydrating: boolean
  isRerunning: boolean
  isClaimHistoryLoading: boolean
  backendError: string | null
  claimHistoryError: string | null
  needsFreshReview: boolean
  freshness: DraftReviewFreshness | null
  latestReviewRun: DraftReviewRunSummary | null
  onPrevious: () => void
  onNext: () => void
  onResolutionAction: (action: ResolutionAction) => void
  onResolutionNoteChange: (note: string) => void
  onProposedClaimTextChange: (text: string) => void
  onSubmitResolution: () => void
  onRetry: () => void
  onRerun: () => void
}) {
  if (draftId == null) {
    return (
      <StatusPane
        title="Draft target required"
        body="Set ?draftId=... in the URL or provide VITE_DEBRIEV_DRAFT_ID before loading the workbench."
      />
    )
  }

  if (isHydrating && claim == null && freshness == null && backendError == null) {
    return <StatusPane title="Loading review state" body="Hydrating the current persisted review queue." loading />
  }

  if (backendError != null && claim == null) {
    return (
      <StatusPane
        title="Review state unavailable"
        body={backendError}
        action={
          <Button variant="outline" size="sm" onClick={onRetry}>
            Retry
          </Button>
        }
      />
    )
  }

  if (claim == null) {
    return (
      <StatusPane
        title="Queue clear"
        body="No active flagged claims remain for this draft. Run review again if the persisted state may be stale."
        action={
          <Button variant="outline" size="sm" onClick={onRerun}>
            Run review
          </Button>
        }
      />
    )
  }

  return (
    <div className="relative flex h-full flex-col bg-background">
      <div className="shrink-0 border-b border-border/15 px-6 py-3">
        <div className="flex items-center justify-between gap-4">
          <div className="text-[11px] text-muted-foreground/55">
            {claim.reviewDisposition === "active"
              ? visibleClaimCount > 0
                ? `${selectedIndex + 1} of ${visibleClaimCount}`
                : "No active claims"
              : "Resolved claim"}
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              className="h-7 px-2.5 text-[11px]"
              disabled={claim.reviewDisposition !== "active" || !hasPrevious}
              onClick={onPrevious}
            >
              Previous
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-7 px-2.5 text-[11px]"
              disabled={claim.reviewDisposition !== "active" || !hasNext}
              onClick={onNext}
            >
              Next
            </Button>
          </div>
        </div>
        {backendError ? <InlineNotice tone="error" className="mt-3">{backendError}</InlineNotice> : null}
        {claimHistoryError ? <InlineNotice tone="error" className="mt-3">{claimHistoryError}</InlineNotice> : null}
        {isRerunning ? <InlineNotice className="mt-3">Fresh review execution in progress. The queue will refresh when the run completes.</InlineNotice> : null}
        {needsFreshReview ? (
          <InlineNotice className="mt-3">
            No persisted fresh review runs exist for this draft yet. Run review to replace fallback persisted claim status with fresh verification results.
          </InlineNotice>
        ) : null}
        {freshness?.isStale ? (
          <InlineNotice className="mt-3">
            Current review state has changed since the last fresh run.
            {freshness.hasDecisionsAfterLatestRun ? " Reviewer decisions were recorded after the last run." : ""}
            {freshness.hasVerificationActivityAfterLatestRun ? " Verification activity was added after the last run." : ""}
          </InlineNotice>
        ) : null}
      </div>

      <ScrollArea className="flex-1 px-4">
        <DocumentReader
          claim={claim}
          claimHistory={claimHistory}
          latestReviewRun={latestReviewRun}
          isHistoryLoading={isClaimHistoryLoading}
        />
      </ScrollArea>

      {claim.reviewDisposition === "active" && resolutionState != null ? (
        <div className="shrink-0 shadow-[0_-4px_24px_rgba(0,0,0,0.02)]">
          <ReviewActionBar
            resolutionState={resolutionState}
            onActionChange={onResolutionAction}
            onNoteChange={onResolutionNoteChange}
            onProposedClaimTextChange={onProposedClaimTextChange}
            onSubmit={onSubmitResolution}
            hasNext={hasNext}
            onNext={onNext}
          />
        </div>
      ) : (
        <div className="shrink-0 border-t border-border/15 px-6 py-3 text-[11px] text-muted-foreground/60">
          Resolved via {formatActionLabel(resolvedClaim?.action ?? claim.latestDecision?.action ?? "acknowledge_risk")}
          {(resolvedClaim?.createdAt ?? claim.latestDecision?.createdAt) ? (
            <> · {formatTimestamp(resolvedClaim?.createdAt ?? claim.latestDecision?.createdAt ?? null)}</>
          ) : null}
        </div>
      )}

      {claim.reviewDisposition === "active" && nextClaim ? (
        <div className="shrink-0 border-t border-border/15 px-6 py-2 text-[11px] text-muted-foreground/50">
          Next in queue: {nextClaim.claimText}
        </div>
      ) : null}
    </div>
  )
}

function StatusPane({
  title,
  body,
  action,
  loading = false,
}: {
  title: string
  body: string
  action?: React.ReactNode
  loading?: boolean
}) {
  return (
    <div className="flex h-full items-center justify-center px-6 py-8">
      <div className="max-w-md text-center">
        {loading ? <LoaderCircle className="mx-auto mb-3 h-5 w-5 animate-spin text-muted-foreground/45" /> : null}
        {!loading ? <AlertCircle className="mx-auto mb-3 h-5 w-5 text-muted-foreground/35" /> : null}
        <div className="font-serif text-xl text-foreground">{title}</div>
        <div className="mt-3 text-sm leading-6 text-muted-foreground">{body}</div>
        {action ? <div className="mt-4">{action}</div> : null}
      </div>
    </div>
  )
}

function InlineNotice({
  children,
  className,
  tone = "default",
}: {
  children: React.ReactNode
  className?: string
  tone?: "default" | "error"
}) {
  return (
    <div
      className={[
        "rounded-md border px-3 py-2 text-[11px] leading-5",
        tone === "error"
          ? "border-destructive/25 bg-destructive/6 text-destructive/80"
          : "border-border/20 bg-surface-2/45 text-muted-foreground/75",
        className ?? "",
      ].join(" ")}
    >
      {children}
    </div>
  )
}

function mergeResolutionStates(
  current: Record<string, ResolutionActionState>,
  workspaceData: ReviewWorkspaceData,
) {
  return Object.fromEntries(
    workspaceData.activeClaims.map((claim) => {
      const detail = workspaceData.detailsById[claim.id]
      const existing = current[claim.id]
      const fallback = buildDefaultResolutionState(detail)

      return [
        claim.id,
        existing
          ? {
              ...fallback,
              ...existing,
              saving: false,
            }
          : fallback,
      ]
    }),
  )
}

function resolveNextSelectedClaimId({
  currentSelectedClaimId,
  preferredClaimId,
  workspaceData,
}: {
  currentSelectedClaimId: string
  preferredClaimId?: string | null
  workspaceData: ReviewWorkspaceData
}) {
  const activeClaimIds = workspaceData.activeClaims.map((claim) => claim.id)
  const resolvedClaimIds = workspaceData.resolvedClaims.map((claim) => claim.id)
  const selectableClaimIds = [...activeClaimIds, ...resolvedClaimIds]

  if (currentSelectedClaimId && selectableClaimIds.includes(currentSelectedClaimId)) {
    return currentSelectedClaimId
  }

  if (preferredClaimId && activeClaimIds.includes(preferredClaimId)) {
    return preferredClaimId
  }

  return activeClaimIds[0] ?? resolvedClaimIds[0] ?? ""
}

function orderClaims(claims: FlaggedClaimListItem[], mode: QueueOrderMode) {
  const ordered = [...claims]

  ordered.sort((left, right) => {
    if (mode === "draft") {
      return left.draftSequence - right.draftSequence
    }

    const verdictDelta = severityRank(left.verdict) - severityRank(right.verdict)
    if (verdictDelta !== 0) {
      return verdictDelta
    }

    return left.draftSequence - right.draftSequence
  })

  return ordered
}

function buildQueueSections(claims: FlaggedClaimListItem[], mode: QueueOrderMode): QueueSection[] {
  if (mode === "draft") {
    return claims.length > 0
      ? [
          {
            key: "draft-order",
            label: "draft order",
            verdict: null,
            claims,
          },
        ]
      : []
  }

  return verdictOrder
    .map((verdict) => ({
      key: verdict,
      label: verdict.replace(/_/g, " "),
      verdict,
      claims: claims.filter((claim) => claim.verdict === verdict),
    }))
    .filter((section) => section.claims.length > 0)
}

function severityRank(verdict: ReviewVerdict) {
  return verdictOrder.indexOf(verdict)
}

function getResolutionValidation(claim: SelectedClaimDetail, resolutionState: ResolutionActionState) {
  const note = resolutionState.draftNote.trim()
  const proposedClaimText = resolutionState.proposedClaimText.trim()

  switch (resolutionState.selectedAction) {
    case "acknowledge_risk":
      return { valid: true }
    case "mark_for_revision":
      return { valid: note.length > 0 }
    case "resolve_with_edit":
      return {
        valid: proposedClaimText.length > 0 && proposedClaimText !== claim.claimText.trim(),
      }
    default:
      return { valid: false }
  }
}

function buildResolutionSubmission(
  claim: SelectedClaimDetail,
  resolutionState: ResolutionActionState,
): ResolutionSubmissionRequest | null {
  const validation = getResolutionValidation(claim, resolutionState)
  if (!validation.valid || resolutionState.selectedAction == null) {
    return null
  }

  return {
    claimId: claim.id,
    action: resolutionState.selectedAction,
    note: resolutionState.draftNote.trim() || null,
    proposedReplacementText:
      resolutionState.selectedAction === "resolve_with_edit"
        ? resolutionState.proposedClaimText.trim()
        : null,
  }
}

function getErrorMessage(error: unknown) {
  if (error instanceof Error && error.message) {
    return error.message
  }

  return "Request failed."
}

function formatActionLabel(action: ResolutionAction) {
  switch (action) {
    case "acknowledge_risk":
      return "acknowledge risk"
    case "mark_for_revision":
      return "mark for revision"
    case "resolve_with_edit":
      return "resolve with edit"
  }
}

function formatTimestamp(value: string | null | undefined) {
  if (!value) {
    return "unknown"
  }

  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return "unknown"
  }

  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  })
}
