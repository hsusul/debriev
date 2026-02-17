const API_BASE = process.env.API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL || "http://api:8000";

type DocumentItem = {
  doc_id: string;
  filename: string;
  created_at: string;
};

async function fetchDocuments(): Promise<DocumentItem[]> {
  const response = await fetch(`${API_BASE}/v1/documents`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Failed to load documents");
  }
  return (await response.json()) as DocumentItem[];
}

export default async function DocumentsPage() {
  let documents: DocumentItem[] = [];
  let error = "";

  try {
    documents = await fetchDocuments();
  } catch (err) {
    error = err instanceof Error ? err.message : "Failed to load documents";
  }

  return (
    <main>
      <h2>Documents</h2>
      {error ? <p style={{ color: "crimson" }}>{error}</p> : null}
      {documents.length === 0 ? (
        <p>No documents found.</p>
      ) : (
        <ul>
          {documents.map((doc) => (
            <li key={doc.doc_id}>
              <a href={`/reports/${doc.doc_id}`}>{doc.filename}</a> ({new Date(doc.created_at).toLocaleString()})
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}
