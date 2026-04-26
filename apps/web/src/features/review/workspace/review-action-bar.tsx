import { Button } from "@/components/ui/button"
import type { ResolutionAction, ResolutionActionState } from "@/features/review/types"
import { ArrowUpRight } from "lucide-react"
import { cn } from "@/lib/utils"

interface ReviewActionBarProps {
  resolutionState: ResolutionActionState
  onActionChange: (action: ResolutionAction) => void
  onNoteChange: (note: string) => void
  onProposedClaimTextChange: (text: string) => void
  onSubmit: () => void
  hasNext: boolean
  onNext: () => void
}

const actionOptions: { value: ResolutionAction; label: string }[] = [
  { value: "resolve_with_edit", label: "Edit" },
  { value: "mark_for_revision", label: "Revise" },
  { value: "acknowledge_risk", label: "Acknowledge" },
]

export function ReviewActionBar({
  resolutionState,
  onActionChange,
  onNoteChange,
  onProposedClaimTextChange,
  onSubmit,
  hasNext,
  onNext,
}: ReviewActionBarProps) {
  const isReady =
    resolutionState.selectedAction === "acknowledge_risk" ||
    (resolutionState.selectedAction === "mark_for_revision" && resolutionState.draftNote.trim().length > 0) ||
    (resolutionState.selectedAction === "resolve_with_edit" && resolutionState.proposedClaimText.trim().length > 0)

  return (
    <div className="border-t border-border/20 bg-surface-1 p-4 w-full relative">
      <div className="mx-auto max-w-3xl space-y-3">
        {/* Expanding input areas */}
        {resolutionState.selectedAction === "resolve_with_edit" && (
          <div className="animate-in fade-in slide-in-from-bottom-1 duration-200">
            <textarea
              autoFocus
              value={resolutionState.proposedClaimText}
              onChange={(e) => onProposedClaimTextChange(e.target.value)}
              className="h-20 w-full resize-none rounded-md border border-border/30 bg-background px-3 py-2.5 text-[13px] leading-relaxed outline-none transition-colors placeholder:text-muted-foreground/40 focus:border-border/60"
              placeholder="Edited claim text..."
            />
          </div>
        )}

        {resolutionState.selectedAction != null && (
          <div className="animate-in fade-in slide-in-from-bottom-1 duration-200">
            <input
              type="text"
              value={resolutionState.draftNote}
              onChange={(e) => onNoteChange(e.target.value)}
              className="flex h-9 w-full rounded-md border border-border/30 bg-background px-3 text-[13px] outline-none transition-colors placeholder:text-muted-foreground/40 focus:border-border/60"
              placeholder={
                resolutionState.selectedAction === "resolve_with_edit"
                  ? "Resolution note (optional)..."
                  : "Note or instructions..."
              }
            />
          </div>
        )}

        {/* Action controls */}
        <div className="flex items-center justify-between gap-3">
          {/* Segmented action selector */}
          <div className="inline-flex rounded-md border border-border/25 bg-surface-2/50 p-0.5">
            {actionOptions.map((opt) => (
              <button
                key={opt.value}
                type="button"
                className={cn(
                  "rounded-sm px-3 py-1.5 text-[11px] font-medium transition-colors duration-100",
                  resolutionState.selectedAction === opt.value
                    ? "bg-surface-3 text-foreground shadow-sm"
                    : "text-muted-foreground/60 hover:text-foreground/80",
                )}
                onClick={() => onActionChange(opt.value)}
              >
                {opt.label}
              </button>
            ))}
          </div>

          {/* Submit / skip */}
          <div className="flex items-center gap-2">
            {resolutionState.selectedAction && (
              <Button
                variant="default"
                size="sm"
                onClick={onSubmit}
                disabled={!isReady || resolutionState.saving}
                className="h-8 rounded-md px-5 text-[11px] font-semibold"
              >
                {resolutionState.saving ? "Saving..." : "Submit"}
              </Button>
            )}

            {!resolutionState.selectedAction && hasNext && (
              <button
                type="button"
                onClick={onNext}
                className="flex items-center gap-1.5 text-[11px] text-muted-foreground/50 hover:text-muted-foreground transition-colors"
              >
                Skip
                <ArrowUpRight className="h-3 w-3" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
