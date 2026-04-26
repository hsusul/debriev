import { CheckCircle2, CircleDashed, HelpCircle, ShieldAlert, ShieldCheck, TriangleAlert } from "lucide-react"

import type { ReviewVerdict } from "@/features/review/types"
import { cn } from "@/lib/utils"

const verdictConfig: Record<
  ReviewVerdict,
  {
    label: string
    icon: typeof ShieldCheck
    className: string
  }
> = {
  supported: {
    label: "Supported",
    icon: ShieldCheck,
    className: "bg-verdict-supported/10 text-[hsl(var(--verdict-supported))]",
  },
  partially_supported: {
    label: "Partial",
    icon: CheckCircle2,
    className: "bg-verdict-partial/10 text-[hsl(var(--verdict-partial))]",
  },
  overstated: {
    label: "Overstated",
    icon: TriangleAlert,
    className: "bg-verdict-overstated/10 text-[hsl(var(--verdict-overstated))]",
  },
  ambiguous: {
    label: "Ambiguous",
    icon: HelpCircle,
    className: "bg-verdict-ambiguous/10 text-[hsl(var(--verdict-ambiguous))]",
  },
  unsupported: {
    label: "Unsupported",
    icon: ShieldAlert,
    className: "bg-verdict-unsupported/10 text-[hsl(var(--verdict-unsupported))]",
  },
  unverified: {
    label: "Unverified",
    icon: CircleDashed,
    className: "bg-verdict-unverified/10 text-[hsl(var(--verdict-unverified))]",
  },
}

export function VerdictBadge({ verdict, className }: { verdict: ReviewVerdict; className?: string }) {
  const config = verdictConfig[verdict]
  const Icon = config.icon
  return (
    <span className={cn("inline-flex items-center gap-1 rounded-sm px-1.5 py-0.5 text-[10px] font-semibold", config.className, className)}>
      <Icon className="h-2.5 w-2.5" />
      {config.label}
    </span>
  )
}
