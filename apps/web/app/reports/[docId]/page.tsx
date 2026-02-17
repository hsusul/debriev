import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

import ReportWorkspace from "./ReportWorkspace";

const API_BASE = process.env.API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL || "http://api:8000";

type VerificationDetails = {
  courtlistener_url?: string;
};

type ReportCitation = {
  raw: string;
  start?: number | null;
  end?: number | null;
  context_text?: string | null;
  verification_status?: string | null;
  verification_details?: VerificationDetails | null;
};

type ReportPayload = {
  version: string;
  overall_score: number;
  summary: string;
  citations: ReportCitation[];
  created_at?: string | null;
};

async function fetchReport(docId: string): Promise<ReportPayload> {
  const response = await fetch(`${API_BASE}/v1/reports/${docId}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Failed to load report");
  }
  return (await response.json()) as ReportPayload;
}

export default async function ReportPage({ params }: { params: { docId: string } }) {
  let report: ReportPayload | null = null;
  let error = "";

  try {
    report = await fetchReport(params.docId);
  } catch (err) {
    error = err instanceof Error ? err.message : "Failed to load report";
  }

  return (
    <main>
      <h2>Report {params.docId}</h2>
      {error ? <p style={{ color: "crimson" }}>{error}</p> : null}
      {report ? <ReportWorkspace docId={params.docId} report={report} /> : null}
    </main>
  );
}
