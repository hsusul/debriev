"use client";

import { useMemo, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";

import VerifyButton from "./VerifyButton";

pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

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

const PUBLIC_API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

function normalizeWhitespace(value: string): string {
  return value.replace(/\s+/g, " ").trim().toLowerCase();
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

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function buildCounts(citations: ReportCitation[]) {
  let verified = 0;
  let notFound = 0;
  let errors = 0;
  let unverified = 0;

  for (const citation of citations) {
    const status = citation.verification_status || "unverified";
    if (status === "verified") {
      verified += 1;
    } else if (status === "not_found") {
      notFound += 1;
    } else if (status === "error") {
      errors += 1;
    } else {
      unverified += 1;
    }
  }

  return { verified, notFound, errors, unverified };
}

function highlightText(text: string, allTerms: string[], selectedRaw: string | null): string {
  const terms = Array.from(new Set(allTerms.map((t) => t.trim()).filter((t) => t.length > 0))).sort(
    (a, b) => b.length - a.length
  );
  if (terms.length === 0) {
    return escapeHtml(text);
  }

  const selectedNormalized = selectedRaw ? normalizeWhitespace(selectedRaw) : null;
  const pattern = new RegExp(terms.map((term) => escapeRegex(term)).join("|"), "gi");

  let output = "";
  let cursor = 0;

  for (const match of text.matchAll(pattern)) {
    const idx = match.index ?? 0;
    const matched = match[0];

    output += escapeHtml(text.slice(cursor, idx));

    const selected = selectedNormalized !== null && normalizeWhitespace(matched) === selectedNormalized;
    const className = selected ? "citation-mark citation-mark-selected" : "citation-mark";
    output += `<mark class="${className}">${escapeHtml(matched)}</mark>`;

    cursor = idx + matched.length;
  }

  output += escapeHtml(text.slice(cursor));
  return output;
}

export default function ReportWorkspace({ docId, report }: { docId: string; report: ReportPayload }) {
  const [numPages, setNumPages] = useState(0);
  const [pageText, setPageText] = useState<Record<number, string>>({});
  const [selectedCitationIndex, setSelectedCitationIndex] = useState<number | null>(null);
  const [selectedCitationRaw, setSelectedCitationRaw] = useState<string | null>(null);
  const [selectedPage, setSelectedPage] = useState<number | null>(null);
  const [pdfError, setPdfError] = useState("");
  const [jumpMessage, setJumpMessage] = useState("");
  const pageRefs = useRef<Record<number, HTMLDivElement | null>>({});

  const counts = useMemo(() => buildCounts(report.citations), [report.citations]);
  const highlightTerms = useMemo(() => report.citations.map((citation) => citation.raw), [report.citations]);
  const pdfUrl = `${PUBLIC_API_BASE}/v1/documents/${docId}/pdf`;

  const onDocumentLoadSuccess = async (pdf: any) => {
    setNumPages(pdf.numPages);
    setPdfError("");

    const textByPage: Record<number, string> = {};
    for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
      const page = await pdf.getPage(pageNumber);
      const textContent = await page.getTextContent();
      textByPage[pageNumber] = textContent.items
        .map((item: any) => (typeof item.str === "string" ? item.str : ""))
        .join(" ");
    }
    setPageText(textByPage);
  };

  const jumpToCitation = (citation: ReportCitation, idx: number) => {
    setSelectedCitationIndex(idx);
    setSelectedCitationRaw(citation.raw);

    const target = normalizeWhitespace(citation.raw || "");
    if (!target) {
      return;
    }

    for (let pageNumber = 1; pageNumber <= numPages; pageNumber += 1) {
      const pageValue = pageText[pageNumber];
      if (!pageValue) {
        continue;
      }

      if (normalizeWhitespace(pageValue).includes(target)) {
        setSelectedPage(pageNumber);
        setJumpMessage("");
        pageRefs.current[pageNumber]?.scrollIntoView({ behavior: "smooth", block: "start" });
        return;
      }
    }

    setJumpMessage("Citation not found in PDF text layer.");
  };

  const clearSelection = () => {
    setSelectedCitationIndex(null);
    setSelectedCitationRaw(null);
    setSelectedPage(null);
    setJumpMessage("");
  };

  return (
    <section>
      <VerifyButton docId={docId} />
      <p>
        <strong>Summary:</strong> {report.summary}
      </p>
      <p>
        <strong>Overall Score:</strong> {report.overall_score}
      </p>
      <p>
        <strong>Counts:</strong> Verified {counts.verified} / Not found {counts.notFound} / Errors {counts.errors} /
        Unverified {counts.unverified}
      </p>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(0, 2fr) minmax(320px, 1fr)",
          gap: "16px",
          alignItems: "start",
        }}
      >
        <div style={{ border: "1px solid #ddd", borderRadius: "8px", maxHeight: "75vh", overflow: "auto", padding: "8px" }}>
          <Document file={pdfUrl} onLoadSuccess={onDocumentLoadSuccess} onLoadError={() => setPdfError("Failed to load PDF")}>
            {Array.from({ length: numPages }, (_, index) => {
              const pageNumber = index + 1;
              return (
                <div
                  key={pageNumber}
                  ref={(node) => {
                    pageRefs.current[pageNumber] = node;
                  }}
                  style={{
                    marginBottom: "12px",
                    border: selectedPage === pageNumber ? "2px solid #0b57d0" : "1px solid transparent",
                    borderRadius: "6px",
                    padding: "4px",
                  }}
                >
                  <Page
                    pageNumber={pageNumber}
                    width={760}
                    customTextRenderer={(item: { str: string }) =>
                      highlightText(item.str || "", highlightTerms, selectedCitationRaw)
                    }
                  />
                </div>
              );
            })}
          </Document>
          {pdfError ? <p style={{ color: "crimson" }}>{pdfError}</p> : null}
        </div>

        <aside style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px", maxHeight: "75vh", overflow: "auto" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
            <h3 style={{ margin: 0 }}>Citations</h3>
            <button type="button" onClick={clearSelection}>
              Clear selection
            </button>
          </div>
          {jumpMessage ? <p style={{ color: "#7a5a00" }}>{jumpMessage}</p> : null}

          {report.citations.length === 0 ? (
            <p>No citations found for this document.</p>
          ) : (
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  <th style={{ textAlign: "left", borderBottom: "1px solid #ddd", padding: "8px" }}>Raw</th>
                  <th style={{ textAlign: "left", borderBottom: "1px solid #ddd", padding: "8px" }}>Context</th>
                  <th style={{ textAlign: "left", borderBottom: "1px solid #ddd", padding: "8px" }}>Status</th>
                  <th style={{ textAlign: "left", borderBottom: "1px solid #ddd", padding: "8px" }}>Link</th>
                </tr>
              </thead>
              <tbody>
                {report.citations.map((citation, idx) => (
                  <tr
                    key={`${citation.raw}-${citation.start ?? "na"}-${idx}`}
                    onClick={() => jumpToCitation(citation, idx)}
                    style={{
                      cursor: "pointer",
                      backgroundColor: selectedCitationIndex === idx ? "#eef4ff" : "transparent",
                    }}
                  >
                    <td style={{ borderBottom: "1px solid #f0f0f0", padding: "8px", verticalAlign: "top" }}>{citation.raw}</td>
                    <td style={{ borderBottom: "1px solid #f0f0f0", padding: "8px", verticalAlign: "top" }}>
                      {truncateContext(citation.context_text) || "-"}
                    </td>
                    <td style={{ borderBottom: "1px solid #f0f0f0", padding: "8px", verticalAlign: "top" }}>
                      {citation.verification_status || "unverified"}
                    </td>
                    <td style={{ borderBottom: "1px solid #f0f0f0", padding: "8px", verticalAlign: "top" }}>
                      {citation.verification_details?.courtlistener_url ? (
                        <a href={citation.verification_details.courtlistener_url} target="_blank" rel="noopener">
                          Open
                        </a>
                      ) : (
                        "-"
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </aside>
      </div>

      <details style={{ marginTop: "16px" }}>
        <summary>View raw JSON</summary>
        <pre>{JSON.stringify(report, null, 2)}</pre>
      </details>

      <style jsx global>{`
        .citation-mark {
          background: #fff4a3;
          padding: 0;
        }
        .citation-mark-selected {
          background: #ffd166;
        }
      `}</style>
    </section>
  );
}
