import type { ReactNode } from "react"
import { cn } from "@/lib/utils"

interface WorkspaceLayoutProps {
  leftPane?: ReactNode
  centerPane: ReactNode
  rightPane?: ReactNode
  className?: string
}

export function WorkspaceLayout({
  leftPane,
  centerPane,
  rightPane,
  className,
}: WorkspaceLayoutProps) {
  return (
    <div
      className={cn(
        "grid h-full w-full",
        leftPane && rightPane
          ? "grid-cols-[260px_minmax(0,1fr)_300px] lg:grid-cols-[280px_minmax(0,1fr)_340px] xl:grid-cols-[300px_minmax(0,1fr)_380px]"
          : leftPane
            ? "grid-cols-[280px_minmax(0,1fr)] xl:grid-cols-[300px_minmax(0,1fr)]"
            : rightPane
              ? "grid-cols-[minmax(0,1fr)_320px] xl:grid-cols-[minmax(0,1fr)_380px]"
              : "grid-cols-1",
        className
      )}
    >
      {leftPane ? (
        <section className="flex h-full min-h-0 flex-col overflow-hidden border-r border-border/20 bg-surface-1">
          {leftPane}
        </section>
      ) : null}

      <section className="flex h-full min-h-0 flex-col overflow-hidden bg-background relative z-10">
        {centerPane}
      </section>

      {rightPane ? (
        <section className="flex h-full min-h-0 flex-col overflow-hidden border-l border-border/20 bg-surface-1">
          {rightPane}
        </section>
      ) : null}
    </div>
  )
}
