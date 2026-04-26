import type { ReactNode } from "react"
import { Shield } from "lucide-react"

export function AppLayout({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-screen flex-col bg-background text-foreground antialiased overflow-hidden">
      <header className="flex h-11 shrink-0 items-center justify-between border-b border-border/30 bg-surface-1 px-4 z-30 relative">
        <div className="flex items-center gap-2.5">
          <Shield className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-[13px] font-semibold tracking-tight text-foreground/90">
            Debriev
          </span>
          <span className="text-[11px] text-muted-foreground/40">/</span>
          <span className="text-[12px] font-medium text-muted-foreground">Review</span>
        </div>
        <div className="flex items-center gap-3 text-[11px] text-muted-foreground/50 font-mono">
          <span>v2.1.0-alpha</span>
        </div>
      </header>
      <main className="flex-1 overflow-hidden">
        {children}
      </main>
    </div>
  )
}
