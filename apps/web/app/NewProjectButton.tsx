"use client";

import { Plus } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { cn } from "../lib/utils";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

type NewProjectButtonProps = {
  variant?: "sidebar" | "card";
  className?: string;
};

export default function NewProjectButton({ variant = "sidebar", className }: NewProjectButtonProps) {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const onCreate = async () => {
    const name = window.prompt("Project name");
    if (!name) {
      return;
    }

    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE}/v1/projects`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || "Failed to create project");
      }
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create project");
    } finally {
      setLoading(false);
    }
  };

  const isCard = variant === "card";

  return (
    <div className={cn("space-y-2", className)}>
      <button
        type="button"
        onClick={onCreate}
        disabled={loading}
        className={cn(
          isCard
            ? "group flex aspect-square w-40 flex-col items-start justify-between rounded-2xl border border-slate-700/70 bg-slate-800/60 p-5 text-left"
            : "inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium text-slate-300 hover:bg-slate-800 hover:text-blue-300",
          "transition",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950",
          "disabled:cursor-not-allowed disabled:opacity-60",
        )}
      >
        <span
          className={cn(
            "inline-flex items-center justify-center rounded-lg",
            isCard ? "h-10 w-10 bg-blue-500/20 text-blue-300 group-hover:bg-blue-500/30" : "h-4 w-4 text-blue-300"
          )}
        >
          <Plus className={cn(isCard ? "h-5 w-5" : "h-3.5 w-3.5")} aria-hidden="true" />
        </span>
        <span className={cn(isCard ? "space-y-1" : "")}>
          <span className={cn("block font-semibold", isCard ? "text-base text-slate-100" : "")}>
            {loading ? "Creating..." : "New Project"}
          </span>
          {isCard ? <span className="block text-xs text-slate-400">Create a workspace bucket.</span> : null}
        </span>
      </button>
      {error ? (
        <p className="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-200">{error}</p>
      ) : null}
    </div>
  );
}
