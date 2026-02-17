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
    <div style={{ marginBottom: "12px" }}>
      <button type="button" onClick={onVerify} disabled={loading}>
        {loading ? "Verifying..." : "Verify (stub)"}
      </button>
      {error ? <p style={{ color: "crimson", marginTop: "8px" }}>{error}</p> : null}
    </div>
  );
}
