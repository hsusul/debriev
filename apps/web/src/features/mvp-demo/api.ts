export interface CitationVerificationResult {
  title: string
  summary: {
    total_claims: number
    total_cited_propositions: number
    flagged_citation_count: number
    verdict_counts: Record<string, number>
    authority_status_counts: Record<string, number>
    authority_content_status_counts: Record<string, number>
  }
  citations: CitationVerificationItem[]
}

export interface CitationVerificationItem {
  citation_text: string
  proposition_text: string
  authority_status: string
  authority_match_status: string
  authority_lookup_status?: string
  authority_lookup_provider?: string | null
  authority_lookup_cached?: boolean
  authority_content_status?: string
  proposition_verdict: string
  reasoning?: string | null
  support_snippet?: string | null
  suggested_fix?: string | null
}

export interface CasePdfVerificationResult {
  pdf_text_status: string
  extracted_authority_metadata: {
    case_name: string | null
    reporter_volume: string | null
    reporter_abbreviation: string | null
    first_page: string | null
    court: string | null
    year: number | null
    canonical_citation: string | null
  } | null
  extracted_character_count: number
  page_count: number | null
  extraction_warnings: string[]
  extracted_text_preview: string | null
  citation_match_status: string
  statement_verdict: string
  reasoning?: string | null
  support_snippet?: string | null
  suggested_fix?: string | null
}

export interface PdfCitationVerificationResult {
  pdf_text_status: string
  extracted_character_count: number
  page_count: number | null
  extraction_warnings: string[]
  extracted_text_preview: string | null
  citation_verification: CitationVerificationResult | null
}

export async function verifyCitationDraft(draftText: string): Promise<CitationVerificationResult> {
  return requestJson<CitationVerificationResult>("/api/v1/citation-verification", {
    method: "POST",
    body: JSON.stringify({ draft_text: draftText }),
    headers: { "Content-Type": "application/json" },
  })
}

export async function verifyCitationPdf(payload: {
  pdfFile: File
  title?: string
}): Promise<PdfCitationVerificationResult> {
  const formData = new FormData()
  formData.append("pdf_file", payload.pdfFile)
  if (payload.title?.trim()) {
    formData.append("title", payload.title.trim())
  }

  return requestJson<PdfCitationVerificationResult>("/api/v1/citation-verification/pdf", {
    method: "POST",
    body: formData,
  })
}

export async function verifyCasePdf(payload: {
  pdfFile: File
  statementText: string
  citationText: string
}): Promise<CasePdfVerificationResult> {
  const formData = new FormData()
  formData.append("pdf_file", payload.pdfFile)
  formData.append("statement_text", payload.statementText)
  if (payload.citationText.trim()) {
    formData.append("citation_text", payload.citationText.trim())
  }

  return requestJson<CasePdfVerificationResult>("/api/v1/case-pdf-verification", {
    method: "POST",
    body: formData,
  })
}

async function requestJson<T>(path: string, init: RequestInit): Promise<T> {
  const response = await fetch(`${resolveApiBaseUrl()}${path}`, init)
  if (!response.ok) {
    throw new Error(await buildApiErrorMessage(response))
  }
  return (await response.json()) as T
}

async function buildApiErrorMessage(response: Response) {
  try {
    const payload = (await response.json()) as { detail?: string }
    if (payload.detail) {
      return payload.detail
    }
  } catch {
    // Fall through to a stable generic message.
  }
  return `Request failed (${response.status})`
}

function resolveApiBaseUrl() {
  const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim()
  if (!configuredBaseUrl) {
    return ""
  }
  return configuredBaseUrl.endsWith("/") ? configuredBaseUrl.slice(0, -1) : configuredBaseUrl
}
