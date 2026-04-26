import { AlertCircle, FileSearch, FolderTree, Link2Off, Shield } from "lucide-react"

import { InspectorSection } from "@/components/review/inspector-section"
import { ScrollArea } from "@/components/ui/scroll-area"
import type { ClaimReviewHistory, SelectedClaimDetail } from "@/features/review/types"
import { cn } from "@/lib/utils"

export function EvidenceInspector({
  claim,
  claimHistory,
  isLoading,
}: {
  claim: SelectedClaimDetail | null
  claimHistory: ClaimReviewHistory | null
  isLoading: boolean
}) {
  if (claim == null) {
    return (
      <div className="flex h-full flex-col">
        <div className="shrink-0 border-b border-border/20 px-4 py-4">
          <h2 className="text-[13px] font-semibold tracking-tight text-foreground/90">Evidence</h2>
        </div>
        <div className="flex flex-1 items-center justify-center px-6 py-8">
          <div className="max-w-[220px] text-center">
            <FileSearch className="mx-auto mb-3 h-5 w-5 text-muted-foreground/30" />
            <div className="text-[12px] font-medium text-muted-foreground/60">Nothing to inspect</div>
            <div className="mt-1.5 text-[11px] leading-relaxed text-muted-foreground/40">
              Select a claim from the queue to view support evidence.
            </div>
          </div>
        </div>
      </div>
    )
  }

  const latestVerification = claimHistory?.latestVerification
  const supportSnapshot = latestVerification?.supportSnapshot
  const primaryAnchor = supportSnapshot?.providerOutput.primaryAnchor ?? claim.primaryAnchor
  const supportAssessments = supportSnapshot?.providerOutput.supportAssessments ?? claim.supportAssessments
  const supportItems = supportSnapshot?.supportItems ?? []
  const includedLinks = supportSnapshot?.validSupportLinks ?? []
  const excludedLinks =
    supportSnapshot?.excludedSupportLinks.map((link) => ({
      code: link.code,
      reason: link.message,
    })) ?? claim.excludedLinks
  const scope = supportSnapshot?.claimScope
    ? {
        scopeKind: supportSnapshot.claimScope.scopeKind,
        allowedSourceCount: supportSnapshot.claimScope.allowedSourceDocumentIds.length,
      }
    : claim.scope

  return (
    <div className="flex h-full flex-col">
      <div className="shrink-0 border-b border-border/20 px-4 py-4">
        <h2 className="text-[13px] font-semibold tracking-tight text-foreground/90">Evidence</h2>
      </div>

      <ScrollArea className="flex-1">
        <div className="space-y-5 px-4 py-4">
          <div className="border-b border-border/15 pb-4">
            <div className="text-[11px] text-muted-foreground/45">Primary anchor</div>
            <div className="mt-3 text-[18px] font-semibold tracking-tight text-foreground">
              {primaryAnchor ?? "No anchor"}
            </div>
            <div className="mt-3 flex flex-wrap gap-1.5">
              {claim.reasoningCategories.map((category) => (
                <span
                  key={category}
                  className="rounded-sm border border-border/15 px-2 py-0.5 text-[10px] text-muted-foreground/60"
                >
                  {category.replace(/_/g, " ")}
                </span>
              ))}
              {claim.contradictionFlags.map((flag) => (
                <span
                  key={flag}
                  className="rounded-sm border border-destructive/20 px-2 py-0.5 text-[10px] text-destructive/70"
                >
                  {flag.replace(/_/g, " ")}
                </span>
              ))}
              {claim.deterministicFlags.map((flag) => (
                <span
                  key={flag}
                  className="rounded-sm border border-border/15 px-2 py-0.5 text-[10px] text-muted-foreground/60"
                >
                  {flag.replace(/_/g, " ")}
                </span>
              ))}
            </div>
            <div className="mt-4 text-[13px] leading-relaxed text-foreground/76">
              {latestVerification?.reasoning ?? claim.evidenceSummary}
            </div>
            <div className="mt-4 flex gap-4 text-[10px] text-muted-foreground/45">
              <span>{includedLinks.length || supportItems.length} included</span>
              <span>{excludedLinks.length} excluded</span>
              <span>{scope?.scopeKind === "bundle" ? "Bundle" : "Matter"}</span>
            </div>
            {latestVerification?.supportSnapshotNote ? (
              <div className="mt-3 text-[11px] leading-relaxed text-muted-foreground/48">
                {latestVerification.supportSnapshotNote}
              </div>
            ) : null}
            {isLoading ? (
              <div className="mt-3 text-[11px] text-muted-foreground/45">Loading claim history.</div>
            ) : null}
          </div>

          <InspectorSection title={`Included support (${Math.max(includedLinks.length, supportItems.length)})`} defaultOpen>
            <div className="space-y-3">
              {supportItems.length > 0 ? (
                supportItems.map((item) => (
                  <div key={`${item.segmentId}-${item.order}`} className="border-b border-border/10 pb-3 last:border-b-0 last:pb-0">
                    <div className="flex items-center justify-between gap-2">
                      <div className="text-[12px] font-semibold text-foreground/85">{item.anchor}</div>
                      <div className="text-[10px] text-muted-foreground/45">{item.segmentType.toLowerCase()}</div>
                    </div>
                    <div className="mt-2">
                      <EvidenceRolePill role={item.evidenceRole} />
                    </div>
                    <div className="mt-2 text-[12px] leading-relaxed text-muted-foreground/72">{item.rawText}</div>
                  </div>
                ))
              ) : includedLinks.length > 0 ? (
                includedLinks.map((link) => (
                  <div key={link.linkId} className="border-b border-border/10 pb-3 last:border-b-0 last:pb-0">
                    <div className="text-[12px] font-semibold text-foreground/85">{link.anchor ?? "Included link"}</div>
                    <div className="mt-2 text-[11px] text-muted-foreground/60">
                      {link.evidenceRole ?? "support"} · {link.linkType.toLowerCase()}
                    </div>
                  </div>
                ))
              ) : (
                <EmptyBlock title="No included support" message="No persisted included support is recorded for this claim." />
              )}
            </div>
          </InspectorSection>

          <InspectorSection title={`Support assessments (${supportAssessments.length})`} defaultOpen={false}>
            <div className="space-y-3">
              {supportAssessments.length > 0 ? (
                supportAssessments.map((item) => (
                  <div key={`${item.segmentId}-${item.anchor}`} className="border-b border-border/10 pb-3 last:border-b-0 last:pb-0">
                    <div className="text-[12px] font-semibold text-foreground/85">{item.anchor ?? "Support item"}</div>
                    <div className="mt-2">
                      <EvidenceRolePill role={item.role} />
                    </div>
                    <div className="mt-2 text-[12px] leading-relaxed text-muted-foreground/72">{item.contribution}</div>
                  </div>
                ))
              ) : (
                <EmptyBlock title="No support assessments" message="No persisted provider-side support assessments are available." />
              )}
            </div>
          </InspectorSection>

          <InspectorSection title={`Excluded (${excludedLinks.length})`} defaultOpen={false}>
            <div className="space-y-3">
              {excludedLinks.length > 0 ? (
                excludedLinks.map((link) => (
                  <div key={`${link.code ?? "excluded"}-${link.reason ?? "reason"}`} className="border-b border-border/10 pb-3 last:border-b-0 last:pb-0">
                    <div className="flex items-center gap-2">
                      <Link2Off className="h-3.5 w-3.5 text-destructive/60" />
                      <div className="text-[12px] font-semibold text-foreground/85">
                        {link.code?.replace(/_/g, " ") ?? "Excluded support"}
                      </div>
                    </div>
                    <div className="mt-2 text-[12px] leading-relaxed text-muted-foreground/72">
                      {link.reason ?? "Excluded from reasoning."}
                    </div>
                  </div>
                ))
              ) : (
                <EmptyBlock title="No exclusions" message="No persisted excluded support is recorded." />
              )}
            </div>
          </InspectorSection>

          <InspectorSection title="Scope" defaultOpen={false}>
            <div className="flex items-start gap-2.5">
              <FolderTree className="mt-0.5 h-3.5 w-3.5 text-muted-foreground/40" />
              <div>
                <div className="text-[12px] font-semibold text-foreground/85">
                  {scope?.scopeKind === "bundle" ? "Evidence bundle" : "Matter fallback"}
                </div>
                <div className="mt-1.5 text-[11px] leading-relaxed text-muted-foreground/60">
                  {scope?.allowedSourceCount ?? 0} allowed source(s)
                </div>
              </div>
            </div>
          </InspectorSection>

          <InspectorSection title={`Related claims (${claim.claimRelationships.length})`} defaultOpen={false}>
            <div className="space-y-3">
              {claim.claimRelationships.length > 0 ? (
                claim.claimRelationships.map((relationship) => (
                  <div
                    key={`${relationship.relationshipType}-${relationship.relatedClaimId}`}
                    className="border-b border-border/10 pb-3 last:border-b-0 last:pb-0"
                  >
                    <div className="text-[12px] font-semibold text-foreground/85">
                      {relationship.relationshipType.replace(/_/g, " ")}
                    </div>
                    <div className="mt-1.5 text-[12px] leading-relaxed text-foreground/75">
                      {relationship.relatedClaimText}
                    </div>
                    {relationship.reasonText ? (
                      <div className="mt-1.5 text-[11px] leading-relaxed text-muted-foreground/60">
                        {relationship.reasonText}
                      </div>
                    ) : null}
                  </div>
                ))
              ) : (
                <EmptyBlock title="No related claims" message="No persisted claim-graph relationships are recorded for this claim." />
              )}
            </div>
          </InspectorSection>
        </div>
      </ScrollArea>
    </div>
  )
}

function EvidenceRolePill({ role }: { role: string }) {
  const normalizedRole = role.toLowerCase()
  const className =
    normalizedRole === "primary"
      ? "border-primary/15 bg-primary/8 text-primary/70"
      : normalizedRole === "secondary"
        ? "border-border/20 bg-surface-3 text-foreground/60"
        : "border-amber-500/15 bg-amber-500/8 text-amber-400/70"

  return (
    <span className={cn("inline-flex items-center gap-1 rounded-sm border px-1.5 py-0.5 text-[10px]", className)}>
      <Shield className="h-2.5 w-2.5" />
      {normalizedRole}
    </span>
  )
}

function EmptyBlock({ title, message }: { title: string; message: string }) {
  return (
    <div className="flex flex-col items-center px-4 py-6 text-center">
      <AlertCircle className="mb-2 h-4 w-4 text-muted-foreground/25" />
      <div className="text-[11px] font-medium text-muted-foreground/50">{title}</div>
      <div className="mt-1 text-[11px] leading-relaxed text-muted-foreground/35">{message}</div>
    </div>
  )
}
