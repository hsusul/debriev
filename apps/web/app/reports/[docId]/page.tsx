import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

import ChatPanel from "../../components/ChatPanel";
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
    <section className="mx-auto w-full max-w-[1400px] space-y-5">
      <div className="rounded-2xl border border-slate-700/70 bg-slate-900/70 px-5 py-4">
        <h2 className="text-xl font-semibold text-slate-100">Report {params.docId}</h2>
        <p className="mt-1 text-sm text-slate-400">Review citations, verification status, and source context.</p>
        {error ? (
          <p className="mt-3 rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-200">{error}</p>
        ) : null}
      </div>

      {report ? <ReportWorkspace docId={params.docId} report={report} /> : null}
      <ChatPanel docId={params.docId} introText="Ask about this report. I will answer from indexed chunks." />
    </section>
  );
}
