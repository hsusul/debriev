import type { ReactNode } from "react"

import { cn } from "@/lib/utils"

export function ReviewPanel({
  title,
  eyebrow,
  meta,
  actions,
  children,
  className,
}: {
  title: string
  eyebrow?: string
  meta?: ReactNode
  actions?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <section className={cn("bg-transparent", className)}>
      <div className="flex items-start justify-between gap-3 border-b border-border/15 px-1 py-2">
        <div className="min-w-0">
          {eyebrow ? (
            <div className="text-[10px] font-mono text-muted-foreground/40">{eyebrow}</div>
          ) : null}
          <h2 className="mt-0.5 text-[14px] font-semibold tracking-tight text-foreground/90">{title}</h2>
          {meta ? <div className="mt-1.5 text-[12px] leading-relaxed text-muted-foreground/60">{meta}</div> : null}
        </div>
        {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
      </div>
      <div className="px-1 py-3">{children}</div>
    </section>
  )
}

export function DenseListRow({
  selected = false,
  children,
  className,
}: {
  selected?: boolean
  children: ReactNode
  className?: string
}) {
  return (
    <div
      className={cn(
        "rounded-sm px-2.5 py-2 transition-colors duration-100",
        selected
          ? "bg-surface-2"
          : "bg-transparent hover:bg-surface-2/50",
        className,
      )}
    >
      {children}
    </div>
  )
}
