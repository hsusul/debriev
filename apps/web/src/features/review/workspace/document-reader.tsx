import { VerdictBadge } from "@/components/review/verdict-badge"
import { Separator } from "@/components/ui/separator"
import type {
  ClaimReviewHistory,
  DraftReviewRunSummary,
  SelectedClaimDetail,
} from "@/features/review/types"

export function DocumentReader({
  claim,
  claimHistory,
  latestReviewRun,
  isHistoryLoading,
}: {
  claim: SelectedClaimDetail
  claimHistory: ClaimReviewHistory | null
  latestReviewRun: DraftReviewRunSummary | null
  isHistoryLoading: boolean
}) {
  const latestVerification = claimHistory?.latestVerification
  const latestDecision = claimHistory?.latestDecision ?? claim.latestDecision

  return (
    <article className="mx-auto max-w-3xl px-8 py-12 lg:py-16 animate-in fade-in duration-300">
      <div className="mb-8 flex items-center justify-between border-b border-border/15 pb-4">
        <VerdictBadge verdict={claim.verdict} className="px-2 py-0.5 text-[11px]" />
        <div className="text-[11px] text-muted-foreground/50">
          {claim.reviewDisposition === "active" ? "Active review" : "Resolved"}
        </div>
      </div>

      <div className="space-y-10">
        <h1 className="font-serif text-2xl font-medium leading-[1.4] tracking-tight text-foreground md:text-3xl md:leading-[1.35]">
          <span className="mr-1.5 select-none text-3xl leading-none text-muted-foreground/15">"</span>
          {claim.claimText}
        </h1>

        <div className="grid gap-3 text-[12px] text-muted-foreground/65 md:grid-cols-3">
          <HistoryMetric
            label="Last fresh review"
            value={latestReviewRun ? formatTimestamp(latestReviewRun.createdAt) : "Not run yet"}
          />
          <HistoryMetric
            label="Latest verification"
            value={latestVerification ? latestVerification.verdict.replace(/_/g, " ") : "None"}
          />
          <HistoryMetric
            label="Latest decision"
            value={latestDecision ? formatDecisionLabel(latestDecision.action) : "Unreviewed"}
          />
        </div>

        {claim.reasoningCategories.length > 0 || claim.contradictionFlags.length > 0 ? (
          <section className="space-y-2">
            <div className="text-[11px] text-muted-foreground/45">Structured reasoning</div>
            <div className="flex flex-wrap gap-2">
              {claim.reasoningCategories.map((category) => (
                <span
                  key={category}
                  className="rounded-sm border border-border/15 px-2 py-0.5 text-[11px] text-muted-foreground/70"
                >
                  {category.replace(/_/g, " ")}
                </span>
              ))}
              {claim.contradictionFlags.map((flag) => (
                <span
                  key={flag}
                  className="rounded-sm border border-destructive/20 px-2 py-0.5 text-[11px] text-destructive/75"
                >
                  {flag.replace(/_/g, " ")}
                </span>
              ))}
            </div>
          </section>
        ) : null}

        {claimHistory?.changeSummary ? (
          <section className="space-y-2">
            <div className="text-[11px] text-muted-foreground/45">Review change</div>
            <div className="text-[13px] leading-relaxed text-foreground/80">
              {buildChangeSummaryText(claimHistory)}
            </div>
          </section>
        ) : null}

        <section className="border-l-2 border-border/25 pl-5">
          <div className="mb-3 text-[11px] text-muted-foreground/45">Draft context</div>
          <p className="text-[14px] leading-relaxed text-muted-foreground/70">
            {claim.claimContext ?? "No persisted assertion context is available for this claim."}
          </p>
        </section>

        <section className="space-y-3">
          <div className="text-[11px] text-muted-foreground/45">
            {claim.reviewDisposition === "active" ? "Why flagged" : "Why it was reviewed"}
          </div>
          <p className="text-[14px] leading-relaxed text-foreground/82">
            {latestVerification?.reasoning ??
              claim.reasoning ??
              "No persisted fresh review reasoning is available yet. Run review to refresh this claim."}
          </p>
        </section>

        {claim.claimRelationships.length > 0 ? (
          <section className="space-y-3">
            <div className="text-[11px] text-muted-foreground/45">Related claims</div>
            <div className="space-y-2">
              {claim.claimRelationships.slice(0, 4).map((relationship) => (
                <div key={`${relationship.relationshipType}-${relationship.relatedClaimId}`} className="border-l border-border/20 pl-3">
                  <div className="text-[12px] font-medium text-foreground/82">
                    {relationship.relationshipType.replace(/_/g, " ")} · {relationship.relatedClaimText}
                  </div>
                  {relationship.reasonText ? (
                    <div className="mt-1 text-[12px] leading-relaxed text-muted-foreground/68">
                      {relationship.reasonText}
                    </div>
                  ) : null}
                </div>
              ))}
            </div>
          </section>
        ) : null}

        {latestDecision ? (
          <>
            <Separator className="bg-border/15" />
            <section className="space-y-2">
              <div className="text-[11px] text-muted-foreground/45">Decision history</div>
              <div className="text-[13px] leading-relaxed text-foreground/82">
                {formatDecisionLabel(latestDecision.action)}
                {latestDecision.createdAt ? ` · ${formatTimestamp(latestDecision.createdAt)}` : ""}
              </div>
              {latestDecision.note ? (
                <div className="text-[13px] leading-relaxed text-muted-foreground/72">{latestDecision.note}</div>
              ) : null}
              {latestDecision.proposedReplacementText ? (
                <div className="rounded-md border border-border/15 px-3 py-2 text-[13px] leading-relaxed text-foreground/78">
                  {latestDecision.proposedReplacementText}
                </div>
              ) : null}
            </section>
          </>
        ) : null}

        {isHistoryLoading ? (
          <div className="text-[12px] text-muted-foreground/45">Loading claim history.</div>
        ) : null}
      </div>
    </article>
  )
}

function HistoryMetric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[11px] text-muted-foreground/45">{label}</div>
      <div className="mt-1 text-[13px] text-foreground/82">{value}</div>
    </div>
  )
}

function buildChangeSummaryText(history: ClaimReviewHistory) {
  const summary = history.changeSummary
  if (summary.verdictChanged && summary.previousVerdict && summary.currentVerdict) {
    return `Verification changed from ${summary.previousVerdict.replace(/_/g, " ")} to ${summary.currentVerdict.replace(/_/g, " ")}.`
  }
  if (summary.flagsChanged && summary.currentFlags.length > 0) {
    return `Deterministic review flags changed. Current flags: ${summary.currentFlags
      .map((flag) => flag.replace(/_/g, " "))
      .join(", ")}.`
  }
  if (summary.supportChanged) {
    return `Evidence posture changed between runs, including anchor/support or excluded-link structure.`
  }
  if (summary.latestDecisionAt && summary.latestAction) {
    return `Latest reviewer action: ${formatDecisionLabel(summary.latestAction)}.`
  }
  if (summary.currentPrimaryAnchor) {
    return `Primary anchor at last review: ${summary.currentPrimaryAnchor}.`
  }
  return "No persisted review changes are recorded for this claim yet."
}

function formatDecisionLabel(action: string) {
  return action.replace(/_/g, " ")
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
