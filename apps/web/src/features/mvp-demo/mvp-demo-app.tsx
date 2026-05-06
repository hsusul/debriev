import type { ReactNode } from "react"
import { useState } from "react"

import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  type CitationVerificationItem,
  type CitationVerificationResult,
  type PdfCitationVerificationResult,
  verifyCitationPdf,
  verifyCitationDraft,
} from "./api"

const sampleDraftText =
  "Citation Memo\n\nBrown v. Board of Education, 347 U.S. 483 (1954), held that segregation in public schools deprives children of equal educational opportunities.\n\nAnderson v. Liberty Lobby, Inc., 477 U.S. 242 (1986), held that summary judgment always fails when facts are disputed."

type DemoMode = "draft" | "pdf"

export function MvpDemoApp() {
  const [mode, setMode] = useState<DemoMode>("draft")
  const [draftText, setDraftText] = useState(sampleDraftText)
  const [citationResult, setCitationResult] = useState<CitationVerificationResult | null>(null)
  const [citationLoading, setCitationLoading] = useState(false)
  const [citationError, setCitationError] = useState<string | null>(null)

  const [draftPdfFile, setDraftPdfFile] = useState<File | null>(null)
  const [pdfCitationResult, setPdfCitationResult] = useState<PdfCitationVerificationResult | null>(null)
  const [pdfLoading, setPdfLoading] = useState(false)
  const [pdfError, setPdfError] = useState<string | null>(null)

  async function runCitationVerification() {
    setCitationLoading(true)
    setCitationError(null)
    try {
      setCitationResult(await verifyCitationDraft(draftText))
    } catch (error) {
      setCitationError(getErrorMessage(error))
    } finally {
      setCitationLoading(false)
    }
  }

  async function runPdfCitationVerification() {
    if (draftPdfFile == null) {
      setPdfError("Upload one text-based draft PDF before running verification.")
      return
    }

    setPdfLoading(true)
    setPdfError(null)
    try {
      setPdfCitationResult(await verifyCitationPdf({ pdfFile: draftPdfFile }))
    } catch (error) {
      setPdfError(getErrorMessage(error))
    } finally {
      setPdfLoading(false)
    }
  }

  return (
    <main className="debriev-backdrop min-h-screen bg-background px-4 py-6 text-foreground md:px-8">
      <div className="mx-auto flex max-w-7xl flex-col gap-6">
        <header className="grid gap-5 border-b border-border/70 pb-5 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-end">
          <div className="max-w-2xl">
            <p className="font-mono text-[11px] uppercase tracking-[0.28em] text-muted-foreground">Debriev MVP</p>
            <h1 className="mt-3 text-2xl font-medium tracking-[-0.04em] text-foreground md:text-3xl">
              Check legal citations
            </h1>
            <p className="mt-2 text-sm leading-6 text-muted-foreground">
              Paste a draft or check one case PDF. Backend statuses are shown directly.
            </p>
          </div>
        </header>

        <Tabs value={mode} onValueChange={(value) => setMode(value as DemoMode)} className="space-y-5">
          <TabsList className="h-auto rounded-none border-b border-border bg-transparent p-0">
            <TabsTrigger
              value="draft"
              className="rounded-none border-b border-transparent bg-transparent px-0 py-3 text-sm text-muted-foreground shadow-none data-[state=active]:border-foreground data-[state=active]:bg-transparent data-[state=active]:text-foreground data-[state=active]:shadow-none"
            >
              Draft citations
            </TabsTrigger>
            <TabsTrigger
              value="pdf"
              className="ml-8 rounded-none border-b border-transparent bg-transparent px-0 py-3 text-sm text-muted-foreground shadow-none data-[state=active]:border-foreground data-[state=active]:bg-transparent data-[state=active]:text-foreground data-[state=active]:shadow-none"
            >
              Draft PDF
            </TabsTrigger>
          </TabsList>

          <TabsContent value="draft" className="mt-0">
            <section className="grid gap-6 xl:grid-cols-[minmax(22rem,0.92fr)_minmax(34rem,1.08fr)]">
              <InputPanel
                endpoint="/api/v1/citation-verification"
                title="Draft"
                action={
                  <Button onClick={runCitationVerification} disabled={citationLoading || !draftText.trim()} size="lg">
                    {citationLoading ? "Verifying..." : "Verify draft"}
                  </Button>
                }
                error={citationError}
              >
                <textarea
                  value={draftText}
                  onChange={(event) => setDraftText(event.target.value)}
                  className="min-h-[31rem] w-full resize-y border-0 border-y border-border/70 bg-transparent px-0 py-5 font-serif text-[18px] leading-8 tracking-[-0.025em] text-foreground outline-none transition placeholder:text-muted-foreground/70 focus:border-ring"
                  spellCheck={false}
                />
                <p className="mt-3 text-xs leading-5 text-muted-foreground">Full case citations only. Short-form references are not resolved yet.</p>
              </InputPanel>

              <CitationResults result={citationResult} loading={citationLoading} />
            </section>
          </TabsContent>

          <TabsContent value="pdf" className="mt-0">
            <section className="grid gap-6 xl:grid-cols-[minmax(22rem,0.92fr)_minmax(34rem,1.08fr)]">
              <InputPanel
                endpoint="/api/v1/citation-verification/pdf"
                title="Draft PDF"
                action={
                  <Button onClick={runPdfCitationVerification} disabled={pdfLoading || draftPdfFile == null} size="lg">
                    {pdfLoading ? "Verifying..." : "Verify PDF"}
                  </Button>
                }
                error={pdfError}
              >
                <label className="group flex min-h-24 cursor-pointer flex-col justify-between border border-dashed border-border bg-surface-0/80 p-4 transition hover:border-ring">
                  <span className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
                    Draft file
                  </span>
                  <span className="mt-5 text-sm text-foreground">{draftPdfFile?.name ?? "Choose one text-based draft PDF"}</span>
                  <span className="mt-1 text-xs text-muted-foreground">The PDF text is extracted, then run through the same citation verifier.</span>
                  <input
                    type="file"
                    accept="application/pdf,.pdf"
                    className="sr-only"
                    onChange={(event) => setDraftPdfFile(event.target.files?.[0] ?? null)}
                  />
                </label>

                <p className="mt-3 text-xs leading-5 text-muted-foreground">No OCR in this pass. Scanned PDFs return extraction diagnostics instead of citation results.</p>
              </InputPanel>

              <PdfCitationResults result={pdfCitationResult} loading={pdfLoading} />
            </section>
          </TabsContent>
        </Tabs>
      </div>
    </main>
  )
}

function InputPanel({
  endpoint,
  title,
  action,
  error,
  children,
}: {
  endpoint: string
  title: string
  action: ReactNode
  error: string | null
  children: ReactNode
}) {
  return (
    <section className="border border-border/80 bg-surface-1/70 p-4 md:p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="font-mono text-[11px] tracking-[0.08em] text-muted-foreground">{endpoint}</p>
          <h2 className="mt-2 text-xl font-medium tracking-[-0.03em] text-foreground">{title}</h2>
        </div>
        {action}
      </div>
      <div className="mt-5">{children}</div>
      {error ? <ErrorBox message={error} /> : null}
    </section>
  )
}

function CitationResults({ result, loading }: { result: CitationVerificationResult | null; loading: boolean }) {
  if (loading) {
    return <EmptyState title="Checking draft" body="Parsing citations and resolving authority identity." />
  }

  if (result == null) {
    return <EmptyState title="No result yet" body="Run verification to inspect citation rows and support status." />
  }

  return (
    <ResultPanel eyebrow={result.title} title="Results">
      <CitationResultBody result={result} />
    </ResultPanel>
  )
}

function CitationRow({ citation }: { citation: CitationVerificationItem }) {
  return (
    <article className="py-5">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <InlineStatus value={citation.proposition_verdict} emphasis />
        <InlineStatus value={citation.authority_match_status} />
        {citation.authority_lookup_status ? <InlineStatus value={citation.authority_lookup_status} /> : null}
        {citation.authority_content_status ? <InlineStatus value={citation.authority_content_status} /> : null}
        {citation.authority_lookup_cached ? <InlineStatus value="cached lookup" /> : null}
      </div>
      <h3 className="mt-4 font-mono text-[15px] leading-6 text-foreground">{citation.citation_text}</h3>
      <p className="mt-3 border-l border-border pl-4 text-sm leading-6 text-muted-foreground">{citation.proposition_text}</p>
      <DetailText label="Reasoning" value={citation.reasoning} />
      <DetailText label="Support snippet" value={citation.support_snippet} />
      <DetailText label="Suggested fix" value={citation.suggested_fix} />
    </article>
  )
}

function PdfCitationResults({ result, loading }: { result: PdfCitationVerificationResult | null; loading: boolean }) {
  if (loading) {
    return <EmptyState title="Checking PDF" body="Extracting draft text and running citation verification." />
  }

  if (result == null) {
    return <EmptyState title="No PDF result" body="Upload a text-based draft PDF to check its citations." />
  }

  return (
    <ResultPanel eyebrow={result.pdf_text_status} title="PDF citations">
      <ExtractionDiagnostics result={result} />

      {result.citation_verification ? (
        <div className="mt-6">
          <CitationResultBody result={result.citation_verification} />
        </div>
      ) : (
        <p className="mt-6 border-l border-border pl-4 text-sm leading-6 text-muted-foreground">
          No citation verification was run because PDF text was not extracted.
        </p>
      )}
    </ResultPanel>
  )
}

function ExtractionDiagnostics({ result }: { result: PdfCitationVerificationResult }) {
  return (
    <section className="border-y border-border/80 py-5">
      <h3 className="text-sm font-medium tracking-[-0.02em] text-foreground">Text extraction</h3>
      <dl className="mt-4 grid gap-4 text-sm sm:grid-cols-3">
        <KeyValue label="Status" value={result.pdf_text_status} />
        <KeyValue label="Characters" value={result.extracted_character_count.toString()} />
        <KeyValue label="Pages" value={result.page_count?.toString() ?? "Unknown"} />
      </dl>
      {result.extraction_warnings.length > 0 ? (
        <div className="mt-4 border-l border-border pl-4">
          <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">Warnings</p>
          <ul className="mt-2 space-y-1 text-sm leading-6 text-muted-foreground">
            {result.extraction_warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </div>
      ) : null}
      {result.extracted_text_preview ? (
        <DetailText label="Extracted text preview" value={result.extracted_text_preview} />
      ) : null}
    </section>
  )
}

function CitationResultBody({ result }: { result: CitationVerificationResult }) {
  return (
    <>
      <div className="grid border-y border-border/80 sm:grid-cols-4">
        <Metric label="Cited propositions" value={result.summary.total_cited_propositions} />
        <Metric label="Flagged" value={result.summary.flagged_citation_count} />
        <Metric label="Authority matched" value={getCount(result.summary.authority_status_counts, ["authority_matched", "linked_authority_support_present"])} />
        <Metric label="Authority unresolved" value={getCount(result.summary.authority_status_counts, ["authority_unverified", "citation_recognized", "authority_candidate_parsed", "not_reviewed"])} />
      </div>

      <div className="mt-5 grid gap-5 border-b border-border/80 pb-5 lg:grid-cols-2">
        <StatusBlock title="Verdicts" counts={result.summary.verdict_counts} />
        <StatusBlock title="Authority" counts={result.summary.authority_status_counts} />
      </div>

      <div className="mt-2 divide-y divide-border/80">
        {result.citations.length === 0 ? (
          <p className="p-5 text-sm text-muted-foreground">No full case citations were emitted for MVP review.</p>
        ) : (
          result.citations.map((citation, index) => <CitationRow key={`${citation.citation_text}-${index}`} citation={citation} />)
        )}
      </div>
    </>
  )
}

function ResultPanel({ eyebrow, title, children }: { eyebrow: string; title: string; children: ReactNode }) {
  return (
    <section className="min-h-[36rem] border border-border/80 bg-surface-1/55 p-4 md:p-5">
      <p className="truncate font-mono text-[11px] tracking-[0.08em] text-muted-foreground">{eyebrow}</p>
      <h2 className="mt-2 text-xl font-medium tracking-[-0.03em] text-foreground">{title}</h2>
      <div className="mt-6">{children}</div>
    </section>
  )
}

function EmptyState({ title, body }: { title: string; body: string }) {
  return (
    <section className="flex min-h-[36rem] flex-col justify-between border border-border/80 bg-surface-1/45 p-5">
      <div>
        <p className="font-mono text-[11px] tracking-[0.08em] text-muted-foreground">Output</p>
        <h2 className="mt-2 text-xl font-medium tracking-[-0.03em] text-foreground">{title}</h2>
      </div>
      <p className="max-w-md text-sm leading-6 text-muted-foreground">{body}</p>
    </section>
  )
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className="p-3 first:pl-0 last:pr-0 sm:border-r sm:border-border/80 sm:last:border-r-0">
      <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground">{label}</p>
      <p className="mt-2 text-2xl font-medium tracking-[-0.04em] text-foreground">{value}</p>
    </div>
  )
}

function StatusBlock({ title, counts }: { title: string; counts: Record<string, number> }) {
  const visibleCounts = Object.entries(counts).filter(([, value]) => value > 0)
  return (
    <section>
      <h3 className="font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground">{title}</h3>
      <div className="mt-3 flex flex-wrap gap-x-5 gap-y-2">
        {visibleCounts.length > 0 ? (
          visibleCounts.map(([key, value]) => <SummaryStatus key={key} label={key} value={value} />)
        ) : (
          <span className="text-xs text-muted-foreground">None</span>
        )}
      </div>
    </section>
  )
}

function SummaryStatus({ label, value }: { label: string; value: number }) {
  const positive = ["authority_found", "authority_matched", "support_verified", "supported"]
  const negative = ["unsupported", "authority_not_found", "authority_name_mismatch", "authority_year_mismatch"]
  const tone = positive.includes(label)
    ? "text-verdict-supported"
    : negative.includes(label)
      ? "text-verdict-unsupported"
      : "text-muted-foreground"

  return (
    <span className={`font-mono text-[12px] uppercase tracking-[0.12em] ${tone}`}>
      {label}: <span className="text-foreground">{value}</span>
    </span>
  )
}

function StatusBadge({ value, emphasis = false, muted = false }: { value: string; emphasis?: boolean; muted?: boolean }) {
  const positive = ["authority_found", "authority_matched", "support_verified", "supported", "citation_matches_pdf", "text_extracted"]
  const negative = ["unsupported", "authority_not_found", "authority_name_mismatch", "authority_year_mismatch", "citation_mismatch", "scanned_or_unreadable"]
  const token = value.split(":")[0]
  const className = muted
    ? "border-border bg-transparent text-muted-foreground/60"
    : positive.includes(token)
      ? "border-verdict-supported/35 bg-verdict-supported/10 text-verdict-supported"
      : negative.includes(token)
        ? "border-verdict-unsupported/35 bg-verdict-unsupported/10 text-verdict-unsupported"
        : emphasis
          ? "border-[#75684f] bg-transparent text-[#d6c39b]"
          : "border-border bg-surface-2 text-muted-foreground"

  return (
    <span className={`inline-flex border px-2.5 py-1 font-mono text-[11px] font-medium uppercase tracking-[0.08em] ${className}`}>
      {value}
    </span>
  )
}

function InlineStatus({ value, emphasis = false }: { value: string; emphasis?: boolean }) {
  const positive = ["authority_found", "authority_matched", "matched", "support_verified", "supported", "SUPPORTED", "citation_matches_pdf", "text_extracted"]
  const negative = ["unsupported", "UNSUPPORTED", "authority_not_found", "authority_name_mismatch", "authority_year_mismatch", "citation_mismatch", "scanned_or_unreadable"]
  const token = value.split(":")[0]
  const tone = positive.includes(token)
    ? "bg-verdict-supported"
    : negative.includes(token)
      ? "bg-verdict-unsupported"
      : emphasis
        ? "bg-[#c9b37d]"
        : "bg-muted-foreground"

  return (
    <span className="inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.08em] text-muted-foreground">
      <span className={`size-1.5 rounded-full ${tone}`} />
      <span>{value}</span>
    </span>
  )
}

function DetailText({ label, value }: { label: string; value?: string | null }) {
  if (!value) {
    return null
  }
  return (
    <section className="mt-5 border-l border-border pl-4">
      <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground">{label}</p>
      <p className="mt-3 text-sm leading-6 text-foreground">{value}</p>
    </section>
  )
}

function KeyValue({ label, value }: { label: string; value?: string | null }) {
  return (
    <div>
      <dt className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">{label}</dt>
      <dd className="mt-1.5 text-sm leading-6 text-foreground">{value || "Not extracted"}</dd>
    </div>
  )
}

function ErrorBox({ message }: { message: string }) {
  return <p className="mt-4 border border-destructive/45 bg-destructive/10 p-3 text-sm leading-6 text-destructive-foreground">{message}</p>
}

function getErrorMessage(error: unknown) {
  const message = error instanceof Error ? error.message : "Request failed"
  if (message === "Failed to fetch") {
    return "Could not reach the API. Confirm the FastAPI server is running on localhost:8000, then refresh this page."
  }
  return message
}

function getCount(counts: Record<string, number>, keys: string[]) {
  return keys.reduce((total, key) => total + (counts[key] ?? 0), 0)
}
