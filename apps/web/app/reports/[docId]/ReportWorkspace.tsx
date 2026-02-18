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
    (a, b) => b.length - a.length,
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
    const className = selected
      ? "rounded bg-blue-300/85 px-0.5 text-slate-900"
      : "rounded bg-blue-500/25 px-0.5 text-blue-100";
    output += `<mark class="${className}">${escapeHtml(matched)}</mark>`;

    cursor = idx + matched.length;
  }

  output += escapeHtml(text.slice(cursor));
  return output;
}

function statusPill(status: string | null | undefined): string {
  const value = status || "unverified";
  if (value === "verified") {
    return "inline-flex rounded-full border border-emerald-400/40 bg-emerald-500/15 px-2 py-0.5 text-xs font-medium text-emerald-200";
  }
  if (value === "not_found") {
    return "inline-flex rounded-full border border-amber-400/40 bg-amber-500/15 px-2 py-0.5 text-xs font-medium text-amber-200";
  }
  if (value === "error") {
    return "inline-flex rounded-full border border-red-400/40 bg-red-500/15 px-2 py-0.5 text-xs font-medium text-red-200";
  }
  return "inline-flex rounded-full border border-slate-600 bg-slate-700/50 px-2 py-0.5 text-xs font-medium text-slate-200";
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
    <section className="space-y-4">
      <div className="rounded-2xl border border-slate-700/70 bg-slate-900/70 p-4">
        <VerifyButton docId={docId} />
        <div className="grid gap-3 text-sm text-slate-200 md:grid-cols-3">
          <p className="rounded-lg border border-slate-700 bg-slate-800/70 p-3">
            <strong className="block text-xs uppercase tracking-wide text-slate-400">Summary</strong>
            <span className="mt-1 block">{report.summary}</span>
          </p>
          <p className="rounded-lg border border-slate-700 bg-slate-800/70 p-3">
            <strong className="block text-xs uppercase tracking-wide text-slate-400">Overall Score</strong>
            <span className="mt-1 block text-lg font-semibold text-blue-200">{report.overall_score}</span>
          </p>
          <p className="rounded-lg border border-slate-700 bg-slate-800/70 p-3">
            <strong className="block text-xs uppercase tracking-wide text-slate-400">Counts</strong>
            <span className="mt-1 block">
              Verified {counts.verified} / Not found {counts.notFound} / Errors {counts.errors} / Unverified {counts.unverified}
            </span>
          </p>
        </div>
      </div>

      <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(320px,1fr)]">
        <div className="max-h-[75vh] overflow-auto rounded-2xl border border-slate-700/70 bg-slate-900/70 p-2">
          <Document file={pdfUrl} onLoadSuccess={onDocumentLoadSuccess} onLoadError={() => setPdfError("Failed to load PDF")}>
            {Array.from({ length: numPages }, (_, index) => {
              const pageNumber = index + 1;
              return (
                <div
                  key={pageNumber}
                  ref={(node) => {
                    pageRefs.current[pageNumber] = node;
                  }}
                  className={`mb-3 rounded-lg border p-1 ${
                    selectedPage === pageNumber ? "border-blue-400" : "border-transparent"
                  }`}
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
          {pdfError ? (
            <p className="mt-2 rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-200">{pdfError}</p>
          ) : null}
        </div>

        <aside className="max-h-[75vh] overflow-auto rounded-2xl border border-slate-700/70 bg-slate-900/70 p-3">
          <div className="mb-2 flex items-center justify-between gap-3">
            <h3 className="text-sm font-semibold text-slate-100">Citations</h3>
            <button
              type="button"
              onClick={clearSelection}
              className="rounded-md border border-slate-700 bg-slate-800 px-2 py-1 text-xs text-slate-200 hover:bg-slate-700"
            >
              Clear selection
            </button>
          </div>

          {jumpMessage ? (
            <p className="mb-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
              {jumpMessage}
            </p>
          ) : null}

          {report.citations.length === 0 ? (
            <p className="text-sm text-slate-400">No citations found for this document.</p>
          ) : (
            <table className="w-full border-collapse text-sm text-slate-200">
              <thead>
                <tr>
                  <th className="border-b border-slate-700 px-2 py-2 text-left text-xs font-semibold uppercase tracking-wide text-slate-400">
                    Raw
                  </th>
                  <th className="border-b border-slate-700 px-2 py-2 text-left text-xs font-semibold uppercase tracking-wide text-slate-400">
                    Context
                  </th>
                  <th className="border-b border-slate-700 px-2 py-2 text-left text-xs font-semibold uppercase tracking-wide text-slate-400">
                    Status
                  </th>
                  <th className="border-b border-slate-700 px-2 py-2 text-left text-xs font-semibold uppercase tracking-wide text-slate-400">
                    Link
                  </th>
                </tr>
              </thead>
              <tbody>
                {report.citations.map((citation, idx) => (
                  <tr
                    key={`${citation.raw}-${citation.start ?? "na"}-${idx}`}
                    onClick={() => jumpToCitation(citation, idx)}
                    className={`cursor-pointer transition ${
                      selectedCitationIndex === idx ? "bg-blue-500/10" : "hover:bg-slate-800/70"
                    }`}
                  >
                    <td className="border-b border-slate-800 px-2 py-2 align-top">{citation.raw}</td>
                    <td className="border-b border-slate-800 px-2 py-2 align-top">
                      {truncateContext(citation.context_text) || "-"}
                    </td>
                    <td className="border-b border-slate-800 px-2 py-2 align-top">
                      <span className={statusPill(citation.verification_status)}>{citation.verification_status || "unverified"}</span>
                    </td>
                    <td className="border-b border-slate-800 px-2 py-2 align-top">
                      {citation.verification_details?.courtlistener_url ? (
                        <a href={citation.verification_details.courtlistener_url} target="_blank" rel="noopener" className="text-blue-300 hover:text-blue-200">
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

      <details className="rounded-2xl border border-slate-700/70 bg-slate-900/60 p-4 text-sm text-slate-200">
        <summary className="cursor-pointer font-medium text-slate-100">View raw JSON</summary>
        <pre className="mt-3 overflow-auto rounded-lg border border-slate-700 bg-slate-950/70 p-3 text-xs text-slate-300">
          {JSON.stringify(report, null, 2)}
        </pre>
      </details>
    </section>
  );
}
