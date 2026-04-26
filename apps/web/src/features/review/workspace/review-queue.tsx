import { RotateCw } from "lucide-react"

import { VerdictBadge } from "@/components/review/verdict-badge"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import type {
  DraftReviewFreshness,
  DraftReviewIntelligenceSummary,
  DraftReviewRunSummary,
  FlaggedClaimListItem,
  QueueOrderMode,
  ResolutionActionState,
  ResolvedClaimListItem,
  ReviewDisposition,
  ReviewVerdict,
} from "@/features/review/types"
import { cn } from "@/lib/utils"

export type QueueSection = {
  key: string
  label: string
  verdict: ReviewVerdict | null
  claims: FlaggedClaimListItem[]
}

export function ReviewQueue({
  noDraftLoaded = false,
  queueSections,
  resolutionStateById,
  selectedClaimId,
  selectedDisposition,
  selectedIndex,
  visibleClaimCount,
  verdictFilter,
  queueOrderMode,
  verdictCounts,
  resolvedClaims,
  freshness,
  latestReviewRun,
  previousReviewRun,
  intelligenceSummary,
  isHydrating,
  isRerunning,
  onSelectClaim,
  onFilterChange,
  onQueueOrderModeChange,
  onRerun,
}: {
  noDraftLoaded?: boolean
  queueSections: QueueSection[]
  resolutionStateById: Record<string, ResolutionActionState>
  selectedClaimId: string
  selectedDisposition: ReviewDisposition | null
  selectedIndex: number
  visibleClaimCount: number
  verdictFilter: ReviewVerdict | "all"
  queueOrderMode: QueueOrderMode
  verdictCounts: Record<ReviewVerdict, number>
  resolvedClaims: ResolvedClaimListItem[]
  freshness: DraftReviewFreshness | null
  latestReviewRun: DraftReviewRunSummary | null
  previousReviewRun: DraftReviewRunSummary | null
  intelligenceSummary: DraftReviewIntelligenceSummary | null
  isHydrating: boolean
  isRerunning: boolean
  onSelectClaim: (claimId: string) => void
  onFilterChange: (verdict: ReviewVerdict | "all") => void
  onQueueOrderModeChange: (mode: QueueOrderMode) => void
  onRerun: () => void
}) {
  const hasPersistedRuns = freshness?.hasPersistedReviewRuns ?? false
  const stateSourceLabel =
    freshness == null
      ? null
      : freshness.hasPersistedReviewRuns
        ? freshness.stateSource === "fresh_execution"
          ? "Fresh execution"
          : "Persisted state"
        : "No fresh run"
  const runStatusText = buildRunStatusText({
    noDraftLoaded,
    isHydrating,
    freshness,
    latestReviewRun,
  })
  const rerunRecommendation = buildRerunRecommendation({
    freshness,
    isRerunning,
  })

  return (
    <div className="flex h-full flex-col">
      <div className="shrink-0 border-b border-border/20 px-4 py-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h2 className="text-[13px] font-semibold tracking-tight text-foreground/90">Review Queue</h2>
            <div className="mt-2 flex flex-wrap items-center gap-2 text-[10px]">
              {stateSourceLabel ? (
                <span className="rounded-sm border border-border/20 bg-surface-2/35 px-1.5 py-0.5 text-muted-foreground/70">
                  {stateSourceLabel}
                </span>
              ) : null}
              {freshness?.latestReviewRunStatus ? (
                <span className="text-muted-foreground/45">{freshness.latestReviewRunStatus.toLowerCase()}</span>
              ) : null}
              {rerunRecommendation ? (
                <span className="text-muted-foreground/55">{rerunRecommendation}</span>
              ) : null}
            </div>
            <div className="mt-1 text-[11px] text-muted-foreground/55">{runStatusText}</div>
          </div>
          <div className="flex items-center gap-2">
            {visibleClaimCount > 0 ? (
              <span className="text-[11px] font-mono text-muted-foreground/50">
                {selectedIndex + 1}/{visibleClaimCount}
              </span>
            ) : null}
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-7 px-2.5 text-[11px]"
              disabled={noDraftLoaded || isHydrating || isRerunning}
              onClick={onRerun}
            >
              <RotateCw className={cn("mr-1.5 h-3 w-3", isRerunning && "animate-spin")} />
              {isRerunning ? "Refreshing" : hasPersistedRuns ? "Rerun" : "Run"}
            </Button>
          </div>
        </div>

        {latestReviewRun ? (
          <div className="mt-3 border-t border-border/15 pt-3 text-[10px] text-muted-foreground/45">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
              <span>
                {latestReviewRun.remainingFlaggedClaims} active · {latestReviewRun.resolvedFlaggedClaims} resolved
              </span>
              {previousReviewRun ? <span>Previous run {formatTimestamp(previousReviewRun.createdAt)}</span> : null}
              {intelligenceSummary ? (
                <span>
                  {intelligenceSummary.mostUnstableClaimIds.length} unstable ·{" "}
                  {intelligenceSummary.contradictionClaimIds.length} contradiction-linked
                </span>
              ) : null}
            </div>
            {freshness?.latestDecisionAt ? (
              <div className="mt-1">Latest reviewer action {formatTimestamp(freshness.latestDecisionAt)}</div>
            ) : null}
            {freshness?.latestClaimVerificationAt && freshness.latestClaimVerificationAt !== freshness.lastReviewRunAt ? (
              <div className="mt-1">Latest verification activity {formatTimestamp(freshness.latestClaimVerificationAt)}</div>
            ) : null}
          </div>
        ) : null}

        <div className="mt-3">
          <DenseSegmentedControl
            value={queueOrderMode}
            disabled={noDraftLoaded}
            options={[
              { value: "severity" as QueueOrderMode, label: "By severity" },
              { value: "draft" as QueueOrderMode, label: "By draft" },
            ]}
            onChange={(value) => onQueueOrderModeChange(value as QueueOrderMode)}
          />
        </div>

        <div className="mt-2.5 flex items-center gap-0.5">
          <FilterChip
            label="All"
            active={verdictFilter === "all"}
            disabled={noDraftLoaded}
            onClick={() => onFilterChange("all")}
          />
          <FilterChip
            label="Bad"
            count={verdictCounts.unsupported}
            active={verdictFilter === "unsupported"}
            disabled={noDraftLoaded}
            title="Unsupported or likely unsupported claims"
            onClick={() => onFilterChange("unsupported")}
          />
          <FilterChip
            label="Over"
            count={verdictCounts.overstated}
            active={verdictFilter === "overstated"}
            disabled={noDraftLoaded}
            title="Overstated claims"
            onClick={() => onFilterChange("overstated")}
          />
          <FilterChip
            label="Amb"
            count={verdictCounts.ambiguous}
            active={verdictFilter === "ambiguous"}
            disabled={noDraftLoaded}
            title="Ambiguous claims"
            onClick={() => onFilterChange("ambiguous")}
          />
        </div>
      </div>

      <ScrollArea className="flex-1">
        <div className="px-2 py-2">
          {noDraftLoaded ? (
            <QueueIdleState />
          ) : queueSections.length === 0 ? (
            <QueueEmptyState
              filterActive={verdictFilter !== "all"}
              onClearFilter={() => onFilterChange("all")}
            />
          ) : (
            queueSections.map((section) => (
              <div key={section.key} className="mb-1">
                <div className="flex items-center gap-2 px-2 py-1.5">
                  {section.verdict != null ? (
                    <>
                      <VerdictBadge verdict={section.verdict} className="px-1.5 py-0 text-[10px]" />
                      <span className="text-[10px] font-mono text-muted-foreground/50">
                        {section.claims.length}
                      </span>
                    </>
                  ) : (
                    <span className="text-[10px] text-muted-foreground/50">{section.label}</span>
                  )}
                </div>
                <div className="space-y-px">
                  {section.claims.map((claim) => {
                    const selected = claim.id === selectedClaimId
                    const resolutionState = resolutionStateById[claim.id]

                    return (
                      <button
                        key={claim.id}
                        type="button"
                        className="block w-full text-left outline-none group"
                        aria-pressed={selected}
                        onClick={() => onSelectClaim(claim.id)}
                      >
                        <div
                          className={cn(
                            "relative rounded-md px-3 py-2.5 transition-colors duration-100",
                            selected ? "bg-surface-2" : "bg-transparent hover:bg-surface-2/50",
                          )}
                        >
                          {selected ? (
                            <div className="absolute inset-y-1 left-0.5 w-0.5 rounded-full bg-foreground/60" />
                          ) : null}
                          <div className="flex items-center gap-1.5">
                            <VerdictBadge verdict={claim.verdict} className="px-1 py-0 text-[9px]" />
                            <span className="text-[9px] font-mono text-muted-foreground/40">
                              {claim.draftSequence}
                            </span>
                            {resolutionState?.dirty ? (
                              <span className="text-[9px] font-mono text-muted-foreground/30">draft</span>
                            ) : null}
                          </div>
                          <div
                            className={cn(
                              "mt-1.5 text-[12px] leading-[1.45] line-clamp-2",
                              selected
                                ? "text-foreground"
                                : "text-muted-foreground group-hover:text-foreground/80",
                            )}
                          >
                            {claim.claimText}
                          </div>
                          <div className="mt-1.5 flex items-center gap-3 text-[10px] font-mono text-muted-foreground/40">
                            <span>{claim.supportCount} support</span>
                            {claim.changedSinceLastRun ? <span>changed</span> : null}
                            {claim.contradictionFlags.length > 0 ? <span>conflict</span> : null}
                            {claim.deterministicFlags[0] ? (
                              <span>{claim.deterministicFlags[0].replace(/_/g, " ")}</span>
                            ) : null}
                          </div>
                        </div>
                      </button>
                    )
                  })}
                </div>
              </div>
            ))
          )}

          {resolvedClaims.length > 0 ? (
            <div className="mt-2 border-t border-border/15 pt-2">
              <div className="px-2 py-1.5 text-[10px] text-muted-foreground/45">Resolved</div>
              <div className="space-y-px">
                {resolvedClaims.slice(0, 8).map((record) => (
                  <ResolvedQueueRow
                    key={`${record.id}-${record.createdAt}`}
                    record={record}
                    selected={selectedDisposition === "resolved" && selectedClaimId === record.id}
                    onClick={() => onSelectClaim(record.id)}
                  />
                ))}
              </div>
            </div>
          ) : null}
        </div>
      </ScrollArea>
    </div>
  )
}

function DenseSegmentedControl<T extends string>({
  value,
  options,
  disabled = false,
  onChange,
}: {
  value: T
  options: { value: T; label: string }[]
  disabled?: boolean
  onChange: (value: T) => void
}) {
  return (
    <div className={cn("inline-flex rounded-md border border-border/30 bg-surface-2/50 p-0.5", disabled && "opacity-50")}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          disabled={disabled}
          className={cn(
            "rounded-sm px-3 py-1 text-[11px] font-medium transition-colors duration-100",
            value === option.value
              ? "bg-surface-3 text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground/80",
          )}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  )
}

function FilterChip({
  label,
  count,
  active,
  disabled = false,
  title,
  onClick,
}: {
  label: string
  count?: number
  active: boolean
  disabled?: boolean
  title?: string
  onClick: () => void
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      title={title}
      aria-label={title ?? label}
      className={cn(
        "rounded-sm px-2 py-1 text-[11px] font-medium transition-colors duration-100 disabled:pointer-events-none disabled:opacity-50",
        active
          ? "bg-surface-3 text-foreground"
          : "text-muted-foreground/60 hover:bg-surface-2/40 hover:text-muted-foreground",
      )}
      onClick={onClick}
    >
      {label}
      {count != null && count > 0 ? (
        <span className={cn("ml-1 text-[10px] font-mono", active ? "text-foreground/50" : "text-muted-foreground/30")}>
          {count}
        </span>
      ) : null}
    </button>
  )
}

function QueueIdleState() {
  return (
    <div className="px-3 py-5">
      <div className="rounded-md border border-border/15 bg-surface-2/35 px-3 py-3">
        <div className="text-[12px] font-medium text-muted-foreground/68">No draft loaded</div>
        <div className="mt-1.5 text-[11px] leading-relaxed text-muted-foreground/48">
          Create a draft in the reader pane to populate the review queue.
        </div>
      </div>
    </div>
  )
}

function QueueEmptyState({
  filterActive,
  onClearFilter,
}: {
  filterActive: boolean
  onClearFilter: () => void
}) {
  return (
    <div className="px-4 py-8">
      <div className="text-[13px] font-semibold text-foreground/80">Queue clear</div>
      <div className="mt-2 text-[12px] leading-relaxed text-muted-foreground">
        {filterActive ? "No active claims match the current filter." : "No active flagged claims remain."}
      </div>
      {filterActive ? (
        <Button variant="ghost" size="sm" className="mt-3 h-7 text-[11px]" onClick={onClearFilter}>
          Show all
        </Button>
      ) : null}
    </div>
  )
}

function ResolvedQueueRow({
  record,
  selected,
  onClick,
}: {
  record: ResolvedClaimListItem
  selected: boolean
  onClick: () => void
}) {
  return (
    <button type="button" className="block w-full text-left" onClick={onClick} aria-pressed={selected}>
      <div
        className={cn(
          "rounded-md px-3 py-2 transition-colors",
          selected ? "bg-surface-2" : "hover:bg-surface-2/40",
        )}
      >
        <div className="flex items-center gap-2">
          <span className="text-[9px] font-mono text-muted-foreground/40">{record.draftSequence}</span>
          <span className="text-[11px] font-medium text-muted-foreground/60">{formatActionLabel(record.action)}</span>
        </div>
        <div className="mt-1 text-[11px] leading-snug text-muted-foreground/40 line-clamp-1">
          {record.claimText}
        </div>
      </div>
    </button>
  )
}

function formatActionLabel(action: ResolvedClaimListItem["action"]) {
  switch (action) {
    case "acknowledge_risk":
      return "Acknowledged"
    case "mark_for_revision":
      return "Revision"
    case "resolve_with_edit":
      return "Edited"
  }
}

function buildRunStatusText({
  noDraftLoaded,
  isHydrating,
  freshness,
  latestReviewRun,
}: {
  noDraftLoaded: boolean
  isHydrating: boolean
  freshness: DraftReviewFreshness | null
  latestReviewRun: DraftReviewRunSummary | null
}) {
  if (noDraftLoaded) {
    return "No draft loaded"
  }
  if (isHydrating) {
    return "Loading persisted review state."
  }
  if (freshness == null || !freshness.hasPersistedReviewRuns || latestReviewRun == null) {
    return "No fresh review run has been recorded yet."
  }
  return `Last review run ${formatTimestamp(latestReviewRun.createdAt)}`
}

function buildRerunRecommendation({
  freshness,
  isRerunning,
}: {
  freshness: DraftReviewFreshness | null
  isRerunning: boolean
}) {
  if (isRerunning) {
    return "Refreshing review now"
  }
  if (freshness == null) {
    return null
  }
  if (!freshness.hasPersistedReviewRuns) {
    return "Run review to analyze this draft"
  }
  if (freshness.isStale) {
    return "Rerun recommended"
  }
  if (freshness.stateSource === "persisted_read") {
    return "Loaded from persisted queue"
  }
  return null
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
