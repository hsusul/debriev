import { useState } from "react"
import { ChevronDown } from "lucide-react"

import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { cn } from "@/lib/utils"

export function InspectorSection({
  title,
  subtitle,
  eyebrow,
  defaultOpen = true,
  children,
}: {
  title: string
  subtitle?: string
  eyebrow?: string
  defaultOpen?: boolean
  children: React.ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <div className="overflow-hidden">
        <CollapsibleTrigger className="flex w-full items-center justify-between gap-3 py-2 text-left hover:bg-surface-2/30 transition-colors rounded-sm px-1 -mx-1">
          <div>
            {eyebrow ? (
              <div className="text-[9px] font-semibold uppercase tracking-widest text-muted-foreground/40">{eyebrow}</div>
            ) : null}
            <div className={cn("text-[12px] font-semibold text-foreground/80", eyebrow ? "mt-1" : "")}>{title}</div>
            {subtitle ? <div className="mt-1 text-[11px] leading-relaxed text-muted-foreground/50">{subtitle}</div> : null}
          </div>
          <ChevronDown className={cn("h-3.5 w-3.5 text-muted-foreground/30 transition-transform duration-200", open && "rotate-180")} />
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="pt-2 pb-1">{children}</div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  )
}
