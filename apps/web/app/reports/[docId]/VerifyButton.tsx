"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export default function VerifyButton({ docId }: { docId: string }) {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const onVerify = async () => {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_BASE}/v1/verify/${docId}`, {
        method: "POST"
      });

      if (!response.ok) {
        throw new Error("Verify request failed");
      }

      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Verify request failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mb-3">
      <button
        type="button"
        onClick={onVerify}
        disabled={loading}
        className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm font-medium text-slate-200 transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-60"
      >
        {loading ? "Verifying..." : "Verify (stub)"}
      </button>
      {error ? (
        <p className="mt-2 rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-200">{error}</p>
      ) : null}
    </div>
  );
}
