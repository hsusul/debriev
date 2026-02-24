"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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

type BestMatch = {
  case_name: string | null;
  court: string | null;
  year: number | null;
  url: string | null;
  matched_citation: string | null;
};

type CitationVerificationFinding = {
  citation: string;
  status: "verified" | "not_found" | "ambiguous";
  confidence: number;
  best_match: BestMatch | null;
  candidates: BestMatch[];
  explanation: string;
  evidence: string;
  probable_case_name: string | null;
};

type CitationVerificationSummary = {
  total: number;
  verified: number;
  not_found: number;
  ambiguous: number;
};

type VerifyCitationsResponse = {
  findings: CitationVerificationFinding[];
  summary: CitationVerificationSummary;
  citations: string[];
  raw?: Record<string, unknown> | null;
};

type BogusFinding = {
  case_name: string;
  reason_label: string;
  reason_phrase: string;
  evidence: string;
  doc_id?: string | null;
  chunk_id?: string | null;
};

type RiskTotals = {
  verified: number;
  not_found: number;
  ambiguous: number;
  bogus: number;
};

type RiskItem = {
  citation: string;
  status: string;
  bogus_reason: string | null;
  evidence: string;
  probable_case_name: string | null;
  link: string | null;
};

type RiskReport = {
  score: number;
  totals: RiskTotals;
  top_risks: RiskItem[];
};

type VerificationHistoryItem = {
  id: number;
  created_at: string;
  summary: CitationVerificationSummary;
  citations_count: number;
};

type ReportOverviewExportEndpoints = {
  pdf_url: string;
  json_client: boolean;
  markdown_client: boolean;
};

type ExtractedCitationsResponse = {
  citations: string[];
  evidence: Record<string, string>;
  probable_case_name: Record<string, string | null>;
};

type ReportOverviewResponse = {
  doc_id: string;
  extracted_citations: ExtractedCitationsResponse | null;
  latest_verification: VerifyCitationsResponse | null;
  risk_report: RiskReport | null;
  bogus_findings: BogusFinding[];
  verification_history: VerificationHistoryItem[];
  export_endpoints: ReportOverviewExportEndpoints;
};

type VerificationJobCreateResponse = {
  job_id: string;
  status: "queued" | "running" | "done" | "failed";
};

type VerificationJobStatusResponse = {
  job_id: string;
  doc_id: string;
  status: "queued" | "running" | "done" | "failed";
  summary: CitationVerificationSummary | null;
  error_text: string | null;
  result_id: number | null;
};

type StatusFilter = "all" | "verified" | "not_found" | "ambiguous" | "bogus";
type SortMode = "risk_desc" | "citation_asc";

type CitationRow = {
  citation: ReportCitation;
  idx: number;
  finding: CitationVerificationFinding | null;
  relatedBogus: boolean;
  riskItem: RiskItem | null;
  effectiveStatus: string;
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

async function parseApiError(response: Response, fallback: string): Promise<string> {
  const raw = await response.text().catch(() => "");
  let detail = raw.trim();
  try {
    const parsed = JSON.parse(raw) as { detail?: string };
    if (parsed.detail) {
      detail = parsed.detail;
    }
  } catch {
    // Keep response text if body is not JSON.
  }
  const suffix = detail ? ` - ${detail}` : "";
  return `${response.status} ${response.statusText || fallback}${suffix}`;
}

async function fetchReportOverview(docId: string): Promise<ReportOverviewResponse> {
  const response = await fetch(`${PUBLIC_API_BASE}/reports/${docId}/overview`, {
    method: "GET"
  });

  if (!response.ok) {
    throw new Error(await parseApiError(response, "Failed to load report overview"));
  }

  return (await response.json()) as ReportOverviewResponse;
}

async function fetchVerificationResultById(docId: string, resultId: number): Promise<VerifyCitationsResponse> {
  const response = await fetch(`${PUBLIC_API_BASE}/reports/${docId}/verification/${resultId}`, {
    method: "GET"
  });

  if (!response.ok) {
    throw new Error(await parseApiError(response, "Failed to load verification run"));
  }

  return (await response.json()) as VerifyCitationsResponse;
}

async function createVerificationJob(docId: string, payload: { text?: string }): Promise<VerificationJobCreateResponse> {
  const response = await fetch(`${PUBLIC_API_BASE}/reports/${docId}/verification/jobs`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    throw new Error(await parseApiError(response, "Failed to create verification job"));
  }

  return (await response.json()) as VerificationJobCreateResponse;
}

async function fetchVerificationJobStatus(docId: string, jobId: string): Promise<VerificationJobStatusResponse> {
  const response = await fetch(`${PUBLIC_API_BASE}/reports/${docId}/verification/jobs/${jobId}`, {
    method: "GET"
  });

  if (!response.ok) {
    throw new Error(await parseApiError(response, "Failed to fetch verification job status"));
  }

  return (await response.json()) as VerificationJobStatusResponse;
}

function normalizeCitationKey(value: string | null | undefined): string {
  return normalizeWhitespace(value || "").replace(/[^\w\s.]/g, "");
}

function findMatchingFinding(
  citation: ReportCitation,
  findings: CitationVerificationFinding[],
): CitationVerificationFinding | null {
  const rawKey = normalizeCitationKey(citation.raw);
  const contextKey = normalizeCitationKey(citation.context_text || "");

  for (const finding of findings) {
    const citationKey = normalizeCitationKey(finding.citation);
    if (!citationKey) {
      continue;
    }
    if (rawKey === citationKey || rawKey.includes(citationKey) || contextKey.includes(citationKey)) {
      return finding;
    }
  }

  for (const finding of findings) {
    const probable = normalizeCitationKey(finding.probable_case_name);
    if (probable && probable === rawKey) {
      return finding;
    }
  }

  return null;
}

function findMatchingRiskItem(citation: ReportCitation, risks: RiskItem[]): RiskItem | null {
  const rawKey = normalizeCitationKey(citation.raw);
  const contextKey = normalizeCitationKey(citation.context_text || "");

  for (const risk of risks) {
    const citationKey = normalizeCitationKey(risk.citation);
    if (!citationKey) {
      continue;
    }
    if (rawKey === citationKey || rawKey.includes(citationKey) || contextKey.includes(citationKey)) {
      return risk;
    }
  }

  return null;
}

function truncateSnippet(value: string, maxLength = 160): string {
  const cleaned = value.replace(/\s+/g, " ").trim();
  if (cleaned.length <= maxLength) {
    return cleaned;
  }
  return `${cleaned.slice(0, maxLength - 1).trimEnd()}…`;
}

function formatVerificationReport(result: VerifyCitationsResponse): string {
  const lines = [
    "Citation Verification Report",
    `Total citations found: ${result.summary.total}`,
    `Verified: ${result.summary.verified} | Not found: ${result.summary.not_found} | Ambiguous: ${result.summary.ambiguous}`
  ];

  const notFound = result.findings
    .filter((finding) => finding.status === "not_found")
    .sort((a, b) => normalizeCitationKey(a.citation).localeCompare(normalizeCitationKey(b.citation)));
  const ambiguous = result.findings
    .filter((finding) => finding.status === "ambiguous")
    .sort((a, b) => normalizeCitationKey(a.citation).localeCompare(normalizeCitationKey(b.citation)));

  if (notFound.length > 0) {
    lines.push(`Top Not Found (max ${Math.min(8, notFound.length)}):`);
    for (const finding of notFound.slice(0, 8)) {
      const snippet = truncateSnippet(finding.evidence);
      lines.push(snippet ? `- ${finding.citation} — ${snippet}` : `- ${finding.citation}`);
    }
  } else {
    lines.push("Top Not Found: none");
  }

  if (ambiguous.length > 0) {
    lines.push(`Top Ambiguous (max ${Math.min(8, ambiguous.length)}):`);
    for (const finding of ambiguous.slice(0, 8)) {
      const matchBits = [
        finding.best_match?.case_name || "",
        finding.best_match?.year ? String(finding.best_match.year) : "",
        finding.best_match?.court || ""
      ].filter(Boolean);
      const matchInfo = matchBits.length > 0 ? ` (best match: ${matchBits.join(", ")})` : "";
      const snippet = truncateSnippet(finding.evidence);
      lines.push(`${`- ${finding.citation}${matchInfo}`}${snippet ? ` — ${snippet}` : ""}`);
    }
  } else {
    lines.push("Top Ambiguous: none");
  }

  return lines.join("\n");
}

function formatVerificationMarkdown(result: VerifyCitationsResponse): string {
  const notFound = result.findings
    .filter((finding) => finding.status === "not_found")
    .sort((a, b) => normalizeCitationKey(a.citation).localeCompare(normalizeCitationKey(b.citation)));
  const ambiguous = result.findings
    .filter((finding) => finding.status === "ambiguous")
    .sort((a, b) => normalizeCitationKey(a.citation).localeCompare(normalizeCitationKey(b.citation)));

  const lines: string[] = [
    "# Citation Verification Report",
    "",
    `- Total citations found: ${result.summary.total}`,
    `- Verified: ${result.summary.verified}`,
    `- Not found: ${result.summary.not_found}`,
    `- Ambiguous: ${result.summary.ambiguous}`,
    ""
  ];

  lines.push("## Top Not Found");
  if (notFound.length === 0) {
    lines.push("- none");
  } else {
    for (const finding of notFound.slice(0, 8)) {
      const snippet = truncateSnippet(finding.evidence);
      lines.push(snippet ? `- ${finding.citation} — ${snippet}` : `- ${finding.citation}`);
    }
  }
  lines.push("");

  lines.push("## Top Ambiguous");
  if (ambiguous.length === 0) {
    lines.push("- none");
  } else {
    for (const finding of ambiguous.slice(0, 8)) {
      const bestParts = [
        finding.best_match?.case_name || "",
        finding.best_match?.year ? String(finding.best_match.year) : "",
        finding.best_match?.court || ""
      ].filter(Boolean);
      const bestText = bestParts.length > 0 ? ` (best match: ${bestParts.join(", ")})` : "";
      const snippet = truncateSnippet(finding.evidence);
      lines.push(`${`- ${finding.citation}${bestText}`}${snippet ? ` — ${snippet}` : ""}`);
    }
  }

  return lines.join("\n");
}

function downloadTextFile(filename: string, content: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
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
  if (value === "ambiguous") {
    return "inline-flex rounded-full border border-blue-400/40 bg-blue-500/15 px-2 py-0.5 text-xs font-medium text-blue-200";
  }
  if (value === "bogus") {
    return "inline-flex rounded-full border border-orange-400/40 bg-orange-500/15 px-2 py-0.5 text-xs font-medium text-orange-200";
  }
  if (value === "error") {
    return "inline-flex rounded-full border border-red-400/40 bg-red-500/15 px-2 py-0.5 text-xs font-medium text-red-200";
  }
  return "inline-flex rounded-full border border-slate-600 bg-slate-700/50 px-2 py-0.5 text-xs font-medium text-slate-200";
}

function statusSeverity(status: string, relatedBogus: boolean): number {
  if (relatedBogus || status === "bogus") {
    return 4;
  }
  if (status === "not_found") {
    return 3;
  }
  if (status === "ambiguous") {
    return 2;
  }
  if (status === "verified") {
    return 1;
  }
  return 0;
}

function formatHistoryTimestamp(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toISOString().replace("T", " ").replace("Z", " UTC");
}

export default function ReportWorkspace({ docId, report }: { docId: string; report: ReportPayload }) {
  const [numPages, setNumPages] = useState(0);
  const [pageText, setPageText] = useState<Record<number, string>>({});
  const [selectedCitationIndex, setSelectedCitationIndex] = useState<number | null>(null);
  const [selectedCitationRaw, setSelectedCitationRaw] = useState<string | null>(null);
  const [selectedPage, setSelectedPage] = useState<number | null>(null);
  const [pdfError, setPdfError] = useState("");
  const [jumpMessage, setJumpMessage] = useState("");
  const [overview, setOverview] = useState<ReportOverviewResponse | null>(null);
  const [verification, setVerification] = useState<VerifyCitationsResponse | null>(null);
  const [verificationBannerError, setVerificationBannerError] = useState("");
  const [bogusFindings, setBogusFindings] = useState<BogusFinding[]>([]);
  const [activeHistoryId, setActiveHistoryId] = useState<number | null>(null);
  const [jobStatusText, setJobStatusText] = useState("");
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [sortMode, setSortMode] = useState<SortMode>("risk_desc");

  const pageRefs = useRef<Record<number, HTMLDivElement | null>>({});

  const loadOverview = useCallback(async () => {
    const data = await fetchReportOverview(docId);
    setOverview(data);
    setVerification(data.latest_verification);
    setBogusFindings(data.bogus_findings);
    setActiveHistoryId(null);
  }, [docId]);

  useEffect(() => {
    let cancelled = false;

    const run = async () => {
      try {
        const data = await fetchReportOverview(docId);
        if (cancelled) {
          return;
        }
        setOverview(data);
        setVerification(data.latest_verification);
        setBogusFindings(data.bogus_findings);
        setActiveHistoryId(null);
      } catch (err) {
        if (!cancelled) {
          const message = err instanceof Error ? err.message : "Failed to load report overview";
          setVerificationBannerError(message);
        }
      }
    };

    void run();
    return () => {
      cancelled = true;
    };
  }, [docId]);

  const riskItems = useMemo(() => overview?.risk_report?.top_risks || [], [overview?.risk_report?.top_risks]);

  const citationRows = useMemo(() => {
    return report.citations.map((citation, idx): CitationRow => {
      const finding = verification ? findMatchingFinding(citation, verification.findings) : null;
      const riskItem = findMatchingRiskItem(citation, riskItems);
      const relatedBogus = bogusFindings.some((item) => {
        const findingKey = normalizeCitationKey(item.case_name);
        const raw = normalizeCitationKey(citation.raw);
        const context = normalizeCitationKey(citation.context_text || "");
        return raw.includes(findingKey) || findingKey.includes(raw) || context.includes(findingKey);
      });
      const effectiveStatus = relatedBogus
        ? "bogus"
        : (finding?.status || citation.verification_status || "unverified");
      return {
        citation,
        idx,
        finding,
        relatedBogus,
        riskItem,
        effectiveStatus
      };
    });
  }, [bogusFindings, report.citations, riskItems, verification]);

  const riskRankByCitation = useMemo(() => {
    const map = new Map<string, number>();
    riskItems.forEach((item, index) => {
      map.set(normalizeCitationKey(item.citation), riskItems.length - index);
    });
    return map;
  }, [riskItems]);

  const filteredRows = useMemo(() => {
    const query = normalizeWhitespace(searchQuery);
    const rows = citationRows.filter((row) => {
      if (statusFilter === "bogus" && !row.relatedBogus) {
        return false;
      }
      if (statusFilter !== "all" && statusFilter !== "bogus" && row.effectiveStatus !== statusFilter) {
        return false;
      }
      if (!query) {
        return true;
      }
      const fields = [
        row.citation.raw,
        row.citation.context_text || "",
        row.finding?.probable_case_name || "",
        row.finding?.citation || ""
      ];
      return fields.some((value) => normalizeWhitespace(value).includes(query));
    });

    return rows.slice().sort((a, b) => {
      if (sortMode === "citation_asc") {
        const left = normalizeCitationKey(a.citation.raw);
        const right = normalizeCitationKey(b.citation.raw);
        const byCitation = left.localeCompare(right);
        if (byCitation !== 0) {
          return byCitation;
        }
        return a.idx - b.idx;
      }

      const leftKey = normalizeCitationKey(a.finding?.citation || a.citation.raw);
      const rightKey = normalizeCitationKey(b.finding?.citation || b.citation.raw);
      const leftRiskRank = riskRankByCitation.get(leftKey) || 0;
      const rightRiskRank = riskRankByCitation.get(rightKey) || 0;
      const leftSeverity = statusSeverity(a.effectiveStatus, a.relatedBogus);
      const rightSeverity = statusSeverity(b.effectiveStatus, b.relatedBogus);
      if (rightSeverity !== leftSeverity) {
        return rightSeverity - leftSeverity;
      }
      if (rightRiskRank !== leftRiskRank) {
        return rightRiskRank - leftRiskRank;
      }
      const byCitation = normalizeCitationKey(a.citation.raw).localeCompare(normalizeCitationKey(b.citation.raw));
      if (byCitation !== 0) {
        return byCitation;
      }
      return a.idx - b.idx;
    });
  }, [citationRows, riskRankByCitation, searchQuery, sortMode, statusFilter]);

  const counts = useMemo(() => {
    if (!verification) {
      return buildCounts(report.citations);
    }
    return {
      verified: verification.summary.verified,
      notFound: verification.summary.not_found,
      errors: 0,
      unverified: Math.max(report.citations.length - verification.summary.total, 0)
    };
  }, [report.citations, verification]);

  const overallScore = useMemo(() => {
    if (overview?.risk_report) {
      return overview.risk_report.score;
    }
    if (!verification) {
      return report.overall_score;
    }
    const total = Math.max(verification.summary.total, 1);
    return Math.round((100 * verification.summary.verified) / total);
  }, [overview?.risk_report, report.overall_score, verification]);

  const verificationReport = useMemo(() => (verification ? formatVerificationReport(verification) : null), [verification]);

  const selectedFinding = useMemo(() => {
    if (selectedCitationIndex === null) {
      return null;
    }
    return citationRows.find((row) => row.idx === selectedCitationIndex)?.finding || null;
  }, [citationRows, selectedCitationIndex]);

  const highlightTerms = useMemo(() => {
    const terms = report.citations.map((citation) => citation.raw);
    if (verification) {
      terms.push(...verification.citations);
    }
    return terms;
  }, [report.citations, verification]);

  const verificationText = useMemo(() => {
    const pieces = report.citations
      .map((citation) => [citation.raw, citation.context_text || ""].filter(Boolean).join("\n"))
      .filter((piece) => piece.trim().length > 0);
    return pieces.join("\n\n");
  }, [report.citations]);

  const verificationHistory = useMemo(() => {
    const rows = overview?.verification_history || [];
    return rows.slice().sort((a, b) => {
      const left = `${a.created_at}|${a.id}`;
      const right = `${b.created_at}|${b.id}`;
      return right.localeCompare(left);
    });
  }, [overview?.verification_history]);

  const exportPdfUrl = useMemo(() => {
    const raw = overview?.export_endpoints?.pdf_url || `/reports/${docId}/export.pdf`;
    if (raw.startsWith("http://") || raw.startsWith("https://")) {
      return raw;
    }
    if (raw.startsWith("/")) {
      return `${PUBLIC_API_BASE}${raw}`;
    }
    return `${PUBLIC_API_BASE}/${raw}`;
  }, [docId, overview?.export_endpoints?.pdf_url]);

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

  const onVerify = async () => {
    setVerificationBannerError("");

    if (!verificationText.trim()) {
      throw new Error("No extracted citation text available to verify.");
    }

    try {
      const created = await createVerificationJob(docId, {
        text: verificationText
      });
      setJobStatusText(`Verification job ${created.job_id} queued...`);

      let finalStatus: VerificationJobStatusResponse | null = null;
      for (let attempt = 0; attempt < 180; attempt += 1) {
        await new Promise((resolve) => {
          setTimeout(resolve, 1000);
        });
        const status = await fetchVerificationJobStatus(docId, created.job_id);
        finalStatus = status;

        if (status.status === "queued") {
          setJobStatusText(`Verification job ${status.job_id} queued...`);
          continue;
        }
        if (status.status === "running") {
          setJobStatusText(`Verification job ${status.job_id} running...`);
          continue;
        }
        if (status.status === "failed") {
          const message = status.error_text || "Verification job failed";
          setJobStatusText("");
          throw new Error(message);
        }
        if (status.status === "done") {
          break;
        }
      }

      if (finalStatus === null || finalStatus.status !== "done") {
        setJobStatusText("");
        throw new Error("Verification job timed out before completion.");
      }

      await loadOverview();
      setJobStatusText("Verification complete.");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Verification request failed";
      setVerificationBannerError(message);
      throw err;
    }
  };

  const selectBogusFinding = (finding: BogusFinding) => {
    setSelectedCitationRaw(finding.case_name);
    const findingKey = normalizeCitationKey(finding.case_name);
    const row = citationRows.find(({ citation }) => {
      const raw = normalizeCitationKey(citation.raw);
      const context = normalizeCitationKey(citation.context_text || "");
      return raw.includes(findingKey) || findingKey.includes(raw) || context.includes(findingKey);
    });

    if (row) {
      jumpToCitation(row.citation, row.idx);
      return;
    }
    setJumpMessage("Related citation row not found for bogus finding.");
  };

  const onSelectHistory = async (resultId: number) => {
    setVerificationBannerError("");
    try {
      const result = await fetchVerificationResultById(docId, resultId);
      setVerification(result);
      setActiveHistoryId(resultId);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to load verification run";
      setVerificationBannerError(message);
    }
  };

  return (
    <section className="space-y-4">
      <div className="rounded-2xl border border-slate-700/70 bg-slate-900/70 p-4">
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <VerifyButton onVerify={onVerify} label="Verify" />
          <button
            type="button"
            onClick={() =>
              verification
                ? downloadTextFile(
                  `verification-${docId}.json`,
                  `${JSON.stringify(verification, null, 2)}\n`,
                  "application/json",
                )
                : null
            }
            disabled={!verification}
            className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm font-medium text-slate-200 transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-60"
          >
            Export JSON
          </button>
          <button
            type="button"
            onClick={() =>
              verification
                ? downloadTextFile(
                  `verification-${docId}.md`,
                  `${formatVerificationMarkdown(verification)}\n`,
                  "text/markdown",
                )
                : null
            }
            disabled={!verification}
            className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm font-medium text-slate-200 transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-60"
          >
            Export Markdown
          </button>
          <a
            href={exportPdfUrl}
            target="_blank"
            rel="noopener"
            className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm font-medium text-slate-200 transition hover:bg-slate-700"
          >
            Export PDF
          </a>
        </div>

        {jobStatusText ? (
          <p className="mb-3 rounded-md border border-blue-500/40 bg-blue-500/10 px-3 py-2 text-sm text-blue-200">
            {jobStatusText}
          </p>
        ) : null}

        {verificationBannerError ? (
          <p className="mb-3 rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-200">
            {verificationBannerError}
          </p>
        ) : null}

        <div className="grid gap-3 text-sm text-slate-200 md:grid-cols-3">
          <p className="rounded-lg border border-slate-700 bg-slate-800/70 p-3">
            <strong className="block text-xs uppercase tracking-wide text-slate-400">Summary</strong>
            <span className="mt-1 block">{report.summary}</span>
          </p>
          <p className="rounded-lg border border-slate-700 bg-slate-800/70 p-3">
            <strong className="block text-xs uppercase tracking-wide text-slate-400">Overall Score</strong>
            <span className="mt-1 block text-lg font-semibold text-blue-200">{overallScore}</span>
          </p>
          <p className="rounded-lg border border-slate-700 bg-slate-800/70 p-3">
            <strong className="block text-xs uppercase tracking-wide text-slate-400">Counts</strong>
            <span className="mt-1 block">
              Verified {counts.verified} / Not found {counts.notFound} / Errors {counts.errors} / Unverified {counts.unverified}
            </span>
          </p>
        </div>
      </div>

      {overview?.risk_report ? (
        <section className="rounded-2xl border border-slate-700/70 bg-slate-900/60 p-4 text-sm text-slate-200">
          <h3 className="text-sm font-semibold text-slate-100">Risk breakdown</h3>
          <div className="mt-3 grid gap-3 md:grid-cols-2">
            <div className="rounded-lg border border-slate-700 bg-slate-800/70 p-3">
              <p className="text-xs uppercase tracking-wide text-slate-400">Score</p>
              <p className="mt-1 text-2xl font-semibold text-blue-200">{overview.risk_report.score}</p>
              <p className="mt-2 text-xs text-slate-400">
                verified {overview.risk_report.totals.verified} / ambiguous {overview.risk_report.totals.ambiguous} / not_found {overview.risk_report.totals.not_found} / bogus {overview.risk_report.totals.bogus}
              </p>
            </div>
            <div className="rounded-lg border border-slate-700 bg-slate-800/70 p-3">
              <p className="text-xs uppercase tracking-wide text-slate-400">Top risks (5)</p>
              <ul className="mt-2 space-y-1 text-xs text-slate-300">
                {overview.risk_report.top_risks.slice(0, 5).map((item, idx) => (
                  <li key={`${item.citation}-${idx}`}>
                    {item.citation} [{item.status}]
                    {item.bogus_reason ? ` (${item.bogus_reason})` : ""}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </section>
      ) : null}

      <section className="rounded-2xl border border-slate-700/70 bg-slate-900/60 p-4 text-sm text-slate-200">
        <h3 className="text-sm font-semibold text-slate-100">History</h3>
        {verificationHistory.length === 0 ? (
          <p className="mt-2 text-xs text-slate-400">No verification history yet.</p>
        ) : (
          <ul className="mt-3 space-y-2">
            {verificationHistory.map((item) => {
              const total = Math.max(item.summary.total, 1);
              const score = Math.round((100 * item.summary.verified) / total);
              const active = activeHistoryId === item.id;
              return (
                <li key={item.id}>
                  <button
                    type="button"
                    onClick={() => void onSelectHistory(item.id)}
                    className={`w-full rounded-lg border px-3 py-2 text-left ${
                      active
                        ? "border-blue-500 bg-blue-500/10"
                        : "border-slate-700 bg-slate-800/60 hover:bg-slate-700/70"
                    }`}
                  >
                    <p className="text-sm text-slate-100">Run #{item.id}</p>
                    <p className="text-xs text-slate-400">{formatHistoryTimestamp(item.created_at)}</p>
                    <p className="mt-1 text-xs text-slate-300">
                      score {score} / total {item.summary.total} / verified {item.summary.verified} / not_found {item.summary.not_found} / ambiguous {item.summary.ambiguous}
                    </p>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </section>

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

          <div className="mb-3 grid gap-2">
            <div className="grid gap-2 sm:grid-cols-2">
              <select
                value={statusFilter}
                onChange={(event) => setStatusFilter(event.target.value as StatusFilter)}
                className="rounded-md border border-slate-700 bg-slate-800 px-2 py-1 text-xs text-slate-200"
              >
                <option value="all">All statuses</option>
                <option value="verified">Verified</option>
                <option value="not_found">Not found</option>
                <option value="ambiguous">Ambiguous</option>
                <option value="bogus">Bogus</option>
              </select>
              <select
                value={sortMode}
                onChange={(event) => setSortMode(event.target.value as SortMode)}
                className="rounded-md border border-slate-700 bg-slate-800 px-2 py-1 text-xs text-slate-200"
              >
                <option value="risk_desc">Sort: risk desc</option>
                <option value="citation_asc">Sort: citation asc</option>
              </select>
            </div>
            <input
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              placeholder="Search citation or probable case name"
              className="rounded-md border border-slate-700 bg-slate-800 px-2 py-1 text-xs text-slate-200 placeholder:text-slate-500"
            />
          </div>

          {jumpMessage ? (
            <p className="mb-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
              {jumpMessage}
            </p>
          ) : null}

          {filteredRows.length === 0 ? (
            <p className="text-sm text-slate-400">No citations matched your filters.</p>
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
                {filteredRows.map((row) => {
                  const { citation, idx, finding, effectiveStatus, relatedBogus } = row;
                  const link = finding?.best_match?.url || citation.verification_details?.courtlistener_url || row.riskItem?.link || null;
                  return (
                    <tr
                      key={`${citation.raw}-${citation.start ?? "na"}-${idx}`}
                      onClick={() => jumpToCitation(citation, idx)}
                      className={`cursor-pointer transition ${
                        selectedCitationIndex === idx
                          ? "bg-blue-500/10"
                          : relatedBogus
                            ? "bg-amber-500/5 hover:bg-amber-500/10"
                            : "hover:bg-slate-800/70"
                      }`}
                    >
                      <td className="border-b border-slate-800 px-2 py-2 align-top">{citation.raw}</td>
                      <td className="border-b border-slate-800 px-2 py-2 align-top">
                        {truncateContext(citation.context_text) || "-"}
                      </td>
                      <td className="border-b border-slate-800 px-2 py-2 align-top">
                        <span className={statusPill(effectiveStatus)}>{effectiveStatus}</span>
                      </td>
                      <td className="border-b border-slate-800 px-2 py-2 align-top">
                        {link ? (
                          <a href={link} target="_blank" rel="noopener" className="text-blue-300 hover:text-blue-200">
                            Open
                          </a>
                        ) : (
                          "-"
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}

          {selectedCitationIndex !== null ? (
            <div className="mt-3 rounded-lg border border-slate-700 bg-slate-800/60 p-3 text-xs text-slate-300">
              <p className="font-semibold uppercase tracking-wide text-slate-400">Citation Details</p>
              <div className="mt-2 space-y-2">
                <p>
                  <span className="text-slate-400">Probable case name:</span>{" "}
                  <span className="text-slate-100">{selectedFinding?.probable_case_name || "-"}</span>
                </p>
                <p>
                  <span className="text-slate-400">Evidence:</span>{" "}
                  <span className="text-slate-100">{selectedFinding?.evidence || "-"}</span>
                </p>
                <p>
                  <span className="text-slate-400">Confidence:</span>{" "}
                  <span className="text-slate-100">
                    {typeof selectedFinding?.confidence === "number" ? selectedFinding.confidence.toFixed(2) : "-"}
                  </span>
                </p>
                <p>
                  <span className="text-slate-400">Best match:</span>{" "}
                  <span className="text-slate-100">
                    {selectedFinding?.best_match?.case_name || "-"}
                    {selectedFinding?.best_match?.court ? ` (${selectedFinding.best_match.court})` : ""}
                    {selectedFinding?.best_match?.year ? ` ${selectedFinding.best_match.year}` : ""}
                  </span>
                </p>
                {selectedFinding?.status === "ambiguous" && selectedFinding.candidates.length > 0 ? (
                  <details className="rounded-md border border-slate-700 bg-slate-900/60 p-2">
                    <summary className="cursor-pointer text-slate-300">
                      View candidates ({selectedFinding.candidates.length})
                    </summary>
                    <ul className="mt-2 space-y-2">
                      {selectedFinding.candidates.map((candidate, idx) => (
                        <li key={`${candidate.case_name || "candidate"}-${idx}`} className="rounded border border-slate-700 px-2 py-1">
                          <p className="text-slate-100">
                            {candidate.case_name || "-"}
                            {candidate.year ? ` (${candidate.year})` : ""}
                          </p>
                          <p className="text-slate-400">{candidate.court || "-"}</p>
                          {candidate.url ? (
                            <a
                              href={candidate.url}
                              target="_blank"
                              rel="noopener"
                              className="text-blue-300 hover:text-blue-200"
                            >
                              {candidate.url}
                            </a>
                          ) : null}
                        </li>
                      ))}
                    </ul>
                  </details>
                ) : null}
              </div>
            </div>
          ) : null}
        </aside>
      </div>

      {verificationReport ? (
        <details open className="rounded-2xl border border-slate-700/70 bg-slate-900/60 p-4 text-sm text-slate-200">
          <summary className="cursor-pointer font-medium text-slate-100">Verification Report</summary>
          <pre className="mt-3 overflow-auto rounded-lg border border-slate-700 bg-slate-950/70 p-3 text-xs text-slate-300">
            {verificationReport}
          </pre>
        </details>
      ) : null}

      <section className="rounded-2xl border border-slate-700/70 bg-slate-900/60 p-4 text-sm text-slate-200">
        <h3 className="text-sm font-semibold text-slate-100">
          Bogus citations ({bogusFindings.length})
        </h3>
        {bogusFindings.length === 0 ? (
          <p className="mt-2 text-xs text-slate-400">No bogus citation findings.</p>
        ) : (
          <ul className="mt-3 space-y-2">
            {bogusFindings.map((finding, idx) => (
              <li key={`${finding.case_name}-${idx}`}>
                <button
                  type="button"
                  onClick={() => selectBogusFinding(finding)}
                  className="w-full rounded-lg border border-slate-700 bg-slate-800/60 px-3 py-2 text-left hover:bg-slate-700/70"
                >
                  <p className="text-sm font-medium text-slate-100">{finding.case_name}</p>
                  <p className="text-xs text-slate-400">
                    {finding.reason_label} ({finding.reason_phrase})
                  </p>
                  <p className="mt-1 text-xs text-slate-300">{finding.evidence || "-"}</p>
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      <details className="rounded-2xl border border-slate-700/70 bg-slate-900/60 p-4 text-sm text-slate-200">
        <summary className="cursor-pointer font-medium text-slate-100">View raw JSON</summary>
        <pre className="mt-3 overflow-auto rounded-lg border border-slate-700 bg-slate-950/70 p-3 text-xs text-slate-300">
          {JSON.stringify(report, null, 2)}
        </pre>
      </details>
    </section>
  );
}
