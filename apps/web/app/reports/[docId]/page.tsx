import VerifyButton from "./VerifyButton";

const API_BASE = process.env.API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL || "http://api:8000";

type ReportCitation = {
  raw: string;
  start?: number | null;
  end?: number | null;
  context_text?: string | null;
  verification_status?: string | null;
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

function truncateContext(value: string | null | undefined): string {
  if (!value) {
    return "";
  }
  if (value.length <= 140) {
    return value;
  }
  return `${value.slice(0, 140)}...`;
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
      {report ? (
        <section>
          <VerifyButton docId={params.docId} />
          <p>
            <strong>Summary:</strong> {report.summary}
          </p>
          <p>
            <strong>Overall Score:</strong> {report.overall_score}
          </p>

          <h3>Citations</h3>
          {report.citations.length === 0 ? (
            <p>No citations found for this document.</p>
          ) : (
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  <th style={{ textAlign: "left", borderBottom: "1px solid #ddd", padding: "8px" }}>Raw</th>
                  <th style={{ textAlign: "left", borderBottom: "1px solid #ddd", padding: "8px" }}>Context</th>
                  <th style={{ textAlign: "left", borderBottom: "1px solid #ddd", padding: "8px" }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {report.citations.map((citation, idx) => (
                  <tr key={`${citation.raw}-${citation.start ?? "na"}-${idx}`}>
                    <td style={{ borderBottom: "1px solid #f0f0f0", padding: "8px", verticalAlign: "top" }}>
                      {citation.raw}
                    </td>
                    <td style={{ borderBottom: "1px solid #f0f0f0", padding: "8px", verticalAlign: "top" }}>
                      {truncateContext(citation.context_text) || "-"}
                    </td>
                    <td style={{ borderBottom: "1px solid #f0f0f0", padding: "8px", verticalAlign: "top" }}>
                      {citation.verification_status || "unverified"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          <details style={{ marginTop: "16px" }}>
            <summary>View raw JSON</summary>
            <pre>{JSON.stringify(report, null, 2)}</pre>
          </details>
        </section>
      ) : null}
    </main>
  );
}
