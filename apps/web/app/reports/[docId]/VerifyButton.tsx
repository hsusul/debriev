"use client";

import { useState } from "react";

type VerifyButtonProps = {
  onVerify: () => Promise<void>;
  label?: string;
};

export default function VerifyButton({ onVerify, label = "Verify" }: VerifyButtonProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const onClick = async () => {
    setLoading(true);
    setError("");
    try {
      await onVerify();
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
        onClick={onClick}
        disabled={loading}
        className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm font-medium text-slate-200 transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-60"
      >
        {loading ? "Verifying..." : label}
      </button>
      {error ? (
        <p className="mt-2 rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-200">{error}</p>
      ) : null}
    </div>
  );
}
