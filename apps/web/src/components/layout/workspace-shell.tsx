import type { ReactNode } from "react"

import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

export function WorkspaceShell({
  heading = "Debriev / Review Workspace",
  title = "Review Workspace",
  subtitle,
  pills = [],
  navigation,
  content,
  inspector,
}: {
  heading?: string
  title?: string
  subtitle?: string
  pills?: string[]
  navigation: ReactNode
  content: ReactNode
  inspector: ReactNode
}) {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto flex min-h-screen max-w-[1700px] flex-col px-3 py-3 md:px-4">
        <header className="mb-3 flex items-center justify-between border-b border-border/80 px-1 pb-3 pt-1">
          <div>
            <div className="text-[11px] text-muted-foreground">{heading}</div>
            <div className="mt-1 font-serif text-xl tracking-[-0.01em] text-foreground">
              {title}
            </div>
            {subtitle ? <div className="mt-1 text-sm text-muted-foreground">{subtitle}</div> : null}
          </div>
          <div className="hidden items-center gap-2 md:flex">
            {pills.map((pill) => (
              <ShellPill key={pill} label={pill} />
            ))}
          </div>
        </header>
        <div className="grid min-h-0 flex-1 gap-3 lg:grid-cols-[300px_minmax(0,1fr)_384px] xl:grid-cols-[320px_minmax(0,1fr)_400px]">
          <WorkspacePane className="lg:min-h-0">{navigation}</WorkspacePane>
          <WorkspacePane className="lg:min-h-0">{content}</WorkspacePane>
          <WorkspacePane className="lg:min-h-0">{inspector}</WorkspacePane>
        </div>
      </div>
    </div>
  )
}

export function WorkspacePane({ className, children }: { className?: string; children: ReactNode }) {
  return (
    <section
      className={cn(
        "min-h-[280px] overflow-hidden rounded-xl border border-border/60 bg-card",
        className,
      )}
    >
      {children}
    </section>
  )
}

function ShellPill({ label }: { label: string }) {
  return (
    <Badge variant="outline" className="rounded-full px-3 py-1 text-[11px] text-muted-foreground">
      {label}
    </Badge>
  )
}
