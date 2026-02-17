const API_BASE = process.env.API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL || "http://api:8000";

async function fetchReport(docId: string): Promise<unknown> {
  const response = await fetch(`${API_BASE}/v1/reports/${docId}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Failed to load report");
  }
  return response.json();
}

export default async function ReportPage({ params }: { params: { docId: string } }) {
  let report: unknown = null;
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
      {report ? <pre>{JSON.stringify(report, null, 2)}</pre> : null}
    </main>
  );
}
