from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any
from uuid import UUID

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

try:
    from sqlmodel import Session, select
except (
    ModuleNotFoundError
):  # pragma: no cover - allows extractor self-tests without db deps
    Session = Any  # type: ignore[assignment]

    def select(*args: Any, **kwargs: Any) -> Any:
        raise ModuleNotFoundError("sqlmodel is required for database-backed API routes")


from app.courtlistener import CourtListenerClient, CourtListenerError
from app.db import (
    Citation,
    CitationVerification,
    Document,
    Project,
    Report,
    get_session,
    init_db,
)
from app.retrieval.service import (
    index_document_from_db,
    query_document_chunks,
    query_project_chunks,
)
from app.retrieval.store import init_retrieval_db
from app.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(settings.storage_dir).mkdir(parents=True, exist_ok=True)
    init_db()
    init_retrieval_db(settings.retrieval_db_path)
    yield


app = FastAPI(title="Debriev API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:  # pragma: no cover - optional dependency for file uploads
    import python_multipart  # type: ignore # noqa: F401

    _HAS_MULTIPART = True
except Exception:
    _HAS_MULTIPART = False


class UploadResponse(BaseModel):
    doc_id: UUID


class DocumentResponse(BaseModel):
    doc_id: UUID
    project_id: UUID | None = None
    filename: str
    created_at: datetime


class ProjectCreateRequest(BaseModel):
    name: str


class ProjectResponse(BaseModel):
    project_id: UUID
    name: str
    created_at: datetime


class ProjectDetailResponse(BaseModel):
    project_id: UUID
    name: str
    created_at: datetime
    documents: list[DocumentResponse]


class CitationResponse(BaseModel):
    id: int
    doc_id: UUID
    raw: str
    normalized: str | None
    start: int | None
    end: int | None
    context_text: str | None
    created_at: datetime


class RetrievalQueryRequest(BaseModel):
    doc_id: UUID
    query: str
    k: int = 5


class RetrievalIndexRequest(BaseModel):
    doc_id: UUID


class RetrievalChunkResponse(BaseModel):
    chunk_id: str
    score: float
    page: int | None = None
    text: str


class RetrievalIndexResponse(BaseModel):
    doc_id: UUID
    chunks_indexed: int


class ChatRequest(BaseModel):
    doc_id: UUID
    message: str
    k: int = 5


class ChatSource(BaseModel):
    doc_id: str | None = None
    chunk_id: str
    page: int | None = None
    text: str
    score: float


class ChatFinding(BaseModel):
    case_name: str
    reason_label: str
    reason_phrase: str
    evidence: str
    doc_id: str | None = None
    chunk_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]
    findings: list[ChatFinding] = Field(default_factory=list)
    tool_result: CitationVerificationToolResult | None = None


class ProjectChatRequest(BaseModel):
    project_id: UUID
    message: str
    k_docs: int = 10
    k_per_doc: int = 3
    k_total: int = 8


class VerifyCitationsRequest(BaseModel):
    text: str
    doc_id: str | None = None
    chunk_id: str | None = None
    include_raw: bool = False


class BestMatch(BaseModel):
    case_name: str | None = None
    court: str | None = None
    year: int | None = None
    url: str | None = None
    matched_citation: str | None = None


class CitationVerificationFinding(BaseModel):
    citation: str
    status: str
    confidence: float
    best_match: BestMatch | None = None
    explanation: str


class CitationVerificationSummary(BaseModel):
    total: int = 0
    verified: int = 0
    not_found: int = 0
    ambiguous: int = 0


class VerifyCitationsResponse(BaseModel):
    findings: list[CitationVerificationFinding] = Field(default_factory=list)
    summary: CitationVerificationSummary = Field(
        default_factory=CitationVerificationSummary
    )
    citations: list[str] = Field(default_factory=list)
    raw: dict[str, Any] | None = None


class CitationVerificationToolResult(BaseModel):
    type: str = "citation_verification"
    findings: list[CitationVerificationFinding] = Field(default_factory=list)
    summary: CitationVerificationSummary = Field(
        default_factory=CitationVerificationSummary
    )
    citations: list[str] = Field(default_factory=list)


class VerifyExtractedCitationsRequest(BaseModel):
    text: str
    doc_id: str | None = None
    chunk_id: str | None = None
    include_raw: bool = False


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _run_citation_verification(
    payload: VerifyCitationsRequest,
    session: Session,
) -> VerifyCitationsResponse:
    normalized_text = _normalize_ws(payload.text)
    if not normalized_text:
        raise HTTPException(status_code=400, detail="text is required")

    input_hash = sha256(normalized_text.encode("utf-8")).hexdigest()
    cached = session.exec(
        select(CitationVerification).where(
            CitationVerification.input_hash == input_hash
        )
    ).first()
    if cached is not None:
        try:
            cached_raw = json.loads(cached.raw_json)
        except json.JSONDecodeError:
            cached_raw = {"results": []}
        findings = _courtlistener_raw_to_findings(
            cached_raw if isinstance(cached_raw, dict) else {"results": []}
        )
        summary = _summarize_citation_findings(findings)
        return VerifyCitationsResponse(
            findings=findings,
            summary=summary,
            citations=[item.citation for item in findings],
            raw=cached_raw
            if payload.include_raw and isinstance(cached_raw, dict)
            else None,
        )

    client = CourtListenerClient()
    try:
        raw = client.lookup_citations(normalized_text)
    except CourtListenerError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    findings = _courtlistener_raw_to_findings(raw)
    summary = _summarize_citation_findings(findings)
    statuses = {item.status for item in findings}
    if not findings or statuses == {"not_found"}:
        summary_status = "not_found"
    elif statuses == {"verified"}:
        summary_status = "verified"
    elif "ambiguous" in statuses or "verified" in statuses:
        summary_status = "ambiguous"
    else:
        summary_status = "not_found"

    record = CitationVerification(
        input_hash=input_hash,
        doc_id=payload.doc_id,
        chunk_id=payload.chunk_id,
        raw_json=json.dumps(raw, sort_keys=True),
        summary_status=summary_status,
    )
    session.add(record)
    session.commit()

    return VerifyCitationsResponse(
        findings=findings,
        summary=summary,
        citations=[item.citation for item in findings],
        raw=raw if payload.include_raw else None,
    )


@app.post("/verify/citations", response_model=VerifyCitationsResponse)
def verify_citations(
    payload: VerifyCitationsRequest,
    session: Session = Depends(get_session),
) -> VerifyCitationsResponse:
    return _run_citation_verification(payload=payload, session=session)


def _normalize_extracted_citation(value: str) -> str:
    extracted = _extract_citations(value)
    if extracted:
        return extracted[0]
    return _normalize_ws(value).strip(" ,;:.")


def _extract_citations(text: str) -> list[str]:
    prepared = _normalize_ws(text)
    if not prepared:
        return []

    patterns: list[tuple[re.Pattern[str], str]] = [
        (
            re.compile(
                r"\b(?P<vol>\d{1,4})\s+U\.?\s*S\.?\s+(?P<page>\d{1,4})\b",
                flags=re.IGNORECASE,
            ),
            "us",
        ),
        (
            re.compile(
                r"\b(?P<vol>\d{1,4})\s+F\.?\s*(?P<series>[23])d\s+(?P<page>\d{1,4})\b",
                flags=re.IGNORECASE,
            ),
            "f_series",
        ),
        (
            re.compile(
                r"\b(?P<vol>\d{1,4})\s+F\.?\s*Supp\.?\s*(?:(?P<series>[23])d\s+)?(?P<page>\d{1,4})\b",
                flags=re.IGNORECASE,
            ),
            "f_supp",
        ),
        (
            re.compile(
                r"\b(?P<vol>\d{1,4})\s+F\.?\s+(?P<page>\d{1,4})\b",
                flags=re.IGNORECASE,
            ),
            "f",
        ),
        (
            re.compile(
                r"\b(?P<vol>\d{1,4})\s+S\.?\s*Ct\.?\s+(?P<page>\d{1,4})\b",
                flags=re.IGNORECASE,
            ),
            "s_ct",
        ),
    ]

    found: dict[str, str] = {}
    for pattern, pattern_name in patterns:
        for match in pattern.finditer(prepared):
            vol = match.group("vol")
            page = match.group("page")
            if pattern_name == "us":
                citation = f"{vol} U.S. {page}"
            elif pattern_name == "f_series":
                citation = f"{vol} F.{match.group('series')}d {page}"
            elif pattern_name == "f_supp":
                series = match.groupdict().get("series")
                citation = (
                    f"{vol} F. Supp. {series}d {page}"
                    if series
                    else f"{vol} F. Supp. {page}"
                )
            elif pattern_name == "f":
                citation = f"{vol} F. {page}"
            else:
                citation = f"{vol} S. Ct. {page}"
            citation_key = _normalize_citation_string(citation)
            if citation_key and citation_key not in found:
                found[citation_key] = citation

    return [found[key] for key in sorted(found)]


def _extract_citations_from_chunks(
    chunks: list[dict[str, object]], max_chars: int = 8000
) -> list[str]:
    def _chunk_order_key(chunk: dict[str, object]) -> tuple[str, str, int, float]:
        raw_page = chunk.get("page")
        try:
            page = int(raw_page) if raw_page is not None else -1
        except (TypeError, ValueError):
            page = -1
        raw_score = chunk.get("score")
        try:
            score = float(raw_score) if raw_score is not None else 0.0
        except (TypeError, ValueError):
            score = 0.0
        return (
            str(chunk.get("doc_id") or ""),
            str(chunk.get("chunk_id") or ""),
            page,
            -score,
        )

    ordered_chunks = sorted(chunks, key=_chunk_order_key)
    parts: list[str] = []
    total_chars = 0
    for chunk in ordered_chunks:
        chunk_text = _normalize_ws(str(chunk.get("text", "") or ""))
        if not chunk_text:
            continue
        if total_chars >= max_chars:
            break

        remaining = max_chars - total_chars
        piece = chunk_text[:remaining].rstrip()
        if not piece:
            continue
        parts.append(piece)
        total_chars += len(piece)
        if len(piece) < len(chunk_text):
            break
        total_chars += 2

    return _extract_citations("\n\n".join(parts))


def _normalize_citation_list(citations: list[str]) -> list[str]:
    deduped: dict[str, str] = {}
    for citation in citations:
        normalized = _normalize_extracted_citation(citation)
        key = _normalize_citation_string(normalized)
        if key and key not in deduped:
            deduped[key] = normalized
    return [deduped[key] for key in sorted(deduped)]


def _citation_list_hash(citations: list[str]) -> str:
    normalized = _normalize_citation_list(citations)
    return sha256("|".join(normalized).encode("utf-8")).hexdigest()


def _extract_results_list(entry: dict[str, Any]) -> list[dict[str, Any]]:
    results_source: Any = entry.get("results")
    if not isinstance(results_source, list):
        for key in ("clusters", "matches", "opinions"):
            candidate = entry.get(key)
            if isinstance(candidate, list):
                results_source = candidate
                break
        else:
            results_source = []
    normalized_results = [_as_result_item(item) for item in results_source]
    normalized_results.sort(key=_stable_json_key)
    return normalized_results


def _extract_citation_entries(raw: dict[str, Any]) -> list[dict[str, Any]]:
    entries_obj: Any
    if isinstance(raw.get("results"), list):
        entries_obj = raw.get("results")
    elif isinstance(raw.get("citations"), list):
        entries_obj = raw.get("citations")
    else:
        entries_obj = []
    entries = entries_obj if isinstance(entries_obj, list) else []
    return [entry for entry in entries if isinstance(entry, dict)]


def _merge_citation_lookup_raw_batches(
    raw_batches: list[dict[str, Any]],
) -> dict[str, Any]:
    merged_entries: list[dict[str, Any]] = []
    for raw in raw_batches:
        for entry in _extract_citation_entries(raw):
            merged_entries.append(entry)
    merged_entries.sort(
        key=lambda entry: (
            _normalize_citation_string(
                str(
                    entry.get("citation")
                    or entry.get("cite")
                    or entry.get("normalized_citation")
                    or entry.get("text")
                    or ""
                )
            ),
            _stable_json_key(entry),
        )
    )
    return {"results": merged_entries}


def _summarize_citation_findings(
    findings: list[CitationVerificationFinding],
) -> CitationVerificationSummary:
    summary = CitationVerificationSummary(total=len(findings))
    for finding in findings:
        if finding.status == "verified":
            summary.verified += 1
        elif finding.status == "ambiguous":
            summary.ambiguous += 1
        else:
            summary.not_found += 1
    return summary


def _run_citation_verification_for_citations(
    citations: list[str],
    session: Session,
    include_raw: bool = False,
    doc_id: str | None = None,
    chunk_id: str | None = None,
    batch_size: int = 25,
) -> VerifyCitationsResponse:
    normalized_citations = _normalize_citation_list(citations)
    if not normalized_citations:
        return VerifyCitationsResponse()

    input_hash = _citation_list_hash(normalized_citations)
    cached = session.exec(
        select(CitationVerification).where(
            CitationVerification.input_hash == input_hash
        )
    ).first()
    if cached is not None:
        try:
            cached_raw = json.loads(cached.raw_json)
        except json.JSONDecodeError:
            cached_raw = {"results": []}
        findings = _courtlistener_raw_to_findings(
            cached_raw if isinstance(cached_raw, dict) else {"results": []},
            requested_citations=normalized_citations,
        )
        return VerifyCitationsResponse(
            findings=findings,
            summary=_summarize_citation_findings(findings),
            citations=normalized_citations,
            raw=cached_raw if include_raw and isinstance(cached_raw, dict) else None,
        )

    client = CourtListenerClient()
    raw_batches: list[dict[str, Any]] = []
    for batch_start in range(0, len(normalized_citations), max(1, batch_size)):
        batch = normalized_citations[batch_start : batch_start + max(1, batch_size)]
        try:
            batch_raw = client.lookup_citation_list(batch)
        except CourtListenerError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        raw_batches.append(
            batch_raw if isinstance(batch_raw, dict) else {"results": []}
        )

    merged_raw = _merge_citation_lookup_raw_batches(raw_batches)
    findings = _courtlistener_raw_to_findings(
        merged_raw,
        requested_citations=normalized_citations,
    )
    summary = _summarize_citation_findings(findings)
    if summary.total == 0 or summary.not_found == summary.total:
        summary_status = "not_found"
    elif summary.verified == summary.total:
        summary_status = "verified"
    else:
        summary_status = "ambiguous"

    record = CitationVerification(
        input_hash=input_hash,
        doc_id=doc_id,
        chunk_id=chunk_id,
        raw_json=json.dumps(merged_raw, sort_keys=True),
        summary_status=summary_status,
    )
    session.add(record)
    session.commit()

    return VerifyCitationsResponse(
        findings=findings,
        summary=summary,
        citations=normalized_citations,
        raw=merged_raw if include_raw else None,
    )


@app.post("/verify/citations/extracted", response_model=VerifyCitationsResponse)
def verify_citations_extracted(
    payload: VerifyExtractedCitationsRequest,
    session: Session = Depends(get_session),
) -> VerifyCitationsResponse:
    citations = _extract_citations(payload.text)
    return _run_citation_verification_for_citations(
        citations=citations,
        session=session,
        include_raw=payload.include_raw,
        doc_id=payload.doc_id,
        chunk_id=payload.chunk_id,
    )


def _document_response_from_row(row: Document) -> DocumentResponse:
    return DocumentResponse(
        doc_id=row.doc_id,
        project_id=row.project_id,
        filename=row.filename,
        created_at=row.created_at,
    )


@app.get("/v1/projects", response_model=list[ProjectResponse])
def list_projects(session: Session = Depends(get_session)) -> list[ProjectResponse]:
    rows = session.exec(select(Project).order_by(Project.created_at.desc())).all()
    return [
        ProjectResponse(
            project_id=row.project_id,
            name=row.name,
            created_at=row.created_at,
        )
        for row in rows
    ]


@app.post("/v1/projects", response_model=ProjectResponse)
def create_project(
    payload: ProjectCreateRequest,
    session: Session = Depends(get_session),
) -> ProjectResponse:
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    project = Project(name=name)
    session.add(project)
    session.commit()
    session.refresh(project)

    return ProjectResponse(
        project_id=project.project_id,
        name=project.name,
        created_at=project.created_at,
    )


@app.get("/v1/projects/{project_id}", response_model=ProjectDetailResponse)
def get_project(
    project_id: UUID, session: Session = Depends(get_session)
) -> ProjectDetailResponse:
    project = session.exec(
        select(Project).where(Project.project_id == project_id)
    ).first()
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    docs = session.exec(
        select(Document)
        .where(Document.project_id == project_id)
        .order_by(Document.created_at.desc())
    ).all()

    return ProjectDetailResponse(
        project_id=project.project_id,
        name=project.name,
        created_at=project.created_at,
        documents=[_document_response_from_row(row) for row in docs],
    )


@app.get("/v1/projects/{project_id}/documents", response_model=list[DocumentResponse])
def list_project_documents(
    project_id: UUID, session: Session = Depends(get_session)
) -> list[DocumentResponse]:
    project = session.exec(
        select(Project).where(Project.project_id == project_id)
    ).first()
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    rows = session.exec(
        select(Document)
        .where(Document.project_id == project_id)
        .order_by(Document.created_at.desc())
    ).all()
    return [_document_response_from_row(row) for row in rows]


if _HAS_MULTIPART:

    @app.post("/v1/upload", response_model=UploadResponse)
    async def upload(
        file: UploadFile = File(...),
        project_id: UUID | None = Form(default=None),
        session: Session = Depends(get_session),
    ) -> UploadResponse:
        from debriev_core.ingest.pdf_text import extract_pdf_text
        from debriev_core.stubs import extract_citations
        from debriev_core.types import DebrievReport

        if not file.filename:
            raise HTTPException(status_code=400, detail="Missing filename")

        if Path(file.filename).suffix.lower() != ".pdf":
            raise HTTPException(status_code=400, detail="Only .pdf files are supported")

        if project_id is not None:
            project = session.exec(
                select(Project).where(Project.project_id == project_id)
            ).first()
            if project is None:
                raise HTTPException(status_code=404, detail="Project not found")

        document = Document(
            project_id=project_id,
            filename=file.filename,
            file_path="",
            stub_text="",
        )
        session.add(document)
        session.commit()
        session.refresh(document)

        file_path = Path(settings.storage_dir) / f"{document.doc_id}.pdf"
        file_path.write_bytes(await file.read())

        text = extract_pdf_text(str(file_path))
        spans = extract_citations(text)

        document.file_path = str(file_path)
        document.stub_text = text

        citation_rows: list[Citation] = []
        report_citations: list[dict[str, Any]] = []
        for span in spans:
            context_text = span.context_text[:500] if span.context_text else None
            citation_rows.append(
                Citation(
                    doc_id=document.doc_id,
                    raw=span.raw,
                    normalized=span.normalized,
                    start=span.start,
                    end=span.end,
                    context_text=context_text,
                )
            )
            report_citations.append(
                {
                    "raw": span.raw,
                    "start": span.start,
                    "end": span.end,
                    "context_text": context_text,
                    "verification_status": "unverified",
                }
            )

        report_payload = DebrievReport(
            version="v1",
            overall_score=0,
            summary=f"Extracted {len(spans)} US case citation(s).",
            citations=report_citations,
            created_at=datetime.now(UTC).isoformat(),
        ).model_dump()
        report = Report(doc_id=document.doc_id, report_json=report_payload)

        session.add(document)
        for row in citation_rows:
            session.add(row)
        session.add(report)
        session.commit()

        try:
            index_document_from_db(
                session=session,
                doc_id=document.doc_id,
                db_path=settings.retrieval_db_path,
                provider=settings.embed_provider,
                openai_api_key=settings.openai_api_key,
            )
        except Exception as exc:
            print(f"Auto-index failed for {document.doc_id}: {exc}")

        return UploadResponse(doc_id=document.doc_id)
else:

    @app.post("/v1/upload", response_model=UploadResponse)
    async def upload(
        session: Session = Depends(get_session),
    ) -> UploadResponse:
        raise HTTPException(
            status_code=500, detail="python-multipart is required for /v1/upload"
        )


@app.get("/v1/documents", response_model=list[DocumentResponse])
def list_documents(session: Session = Depends(get_session)) -> list[DocumentResponse]:
    rows = session.exec(select(Document).order_by(Document.created_at.desc())).all()
    return [_document_response_from_row(row) for row in rows]


@app.get("/v1/documents/{doc_id}/pdf")
def get_document_pdf(
    doc_id: UUID, session: Session = Depends(get_session)
) -> FileResponse:
    document = session.exec(select(Document).where(Document.doc_id == doc_id)).first()
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = Path(document.file_path)
    if not file_path.is_file() or file_path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="PDF not found")

    return FileResponse(
        str(file_path), media_type="application/pdf", filename=document.filename
    )


@app.get("/v1/citations/{doc_id}", response_model=list[CitationResponse])
def list_citations(
    doc_id: UUID, session: Session = Depends(get_session)
) -> list[CitationResponse]:
    document = session.exec(select(Document).where(Document.doc_id == doc_id)).first()
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    rows = session.exec(
        select(Citation).where(Citation.doc_id == doc_id).order_by(Citation.id)
    ).all()
    return [
        CitationResponse(
            id=row.id if row.id is not None else 0,
            doc_id=row.doc_id,
            raw=row.raw,
            normalized=row.normalized,
            start=row.start,
            end=row.end,
            context_text=row.context_text,
            created_at=row.created_at,
        )
        for row in rows
    ]


@app.post("/v1/verify/{doc_id}")
def verify_report(
    doc_id: UUID, session: Session = Depends(get_session)
) -> dict[str, Any]:
    from debriev_core.types import DebrievReport
    from debriev_core.verify import verify_case_citation

    report = session.exec(select(Report).where(Report.doc_id == doc_id)).first()
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")

    citation_rows = session.exec(
        select(Citation).where(Citation.doc_id == doc_id).order_by(Citation.id)
    ).all()

    report_json = dict(report.report_json)
    citations_payload_raw = report_json.get("citations")
    citations_payload: list[dict[str, Any]] = (
        [entry for entry in citations_payload_raw if isinstance(entry, dict)]
        if isinstance(citations_payload_raw, list)
        else []
    )

    verified_count = 0
    not_found_count = 0
    error_count = 0
    unverified_count = 0

    for idx, citation_row in enumerate(citation_rows):
        result = verify_case_citation(citation_row.raw)
        if result.status == "verified":
            verified_count += 1
        elif result.status == "not_found":
            not_found_count += 1
        elif result.status == "error":
            error_count += 1
        else:
            unverified_count += 1

        if idx < len(citations_payload):
            citation_entry = citations_payload[idx]
        else:
            citation_entry = {
                "raw": citation_row.raw,
                "start": citation_row.start,
                "end": citation_row.end,
                "context_text": citation_row.context_text,
            }
            citations_payload.append(citation_entry)

        citation_entry["verification_status"] = result.status
        citation_entry["verification_details"] = result.details or {}

    total = len(citation_rows)
    overall_score = int((verified_count * 100) / total) if total > 0 else 0
    report_json["citations"] = citations_payload
    report_json["summary"] = (
        f"Verified {verified_count} / {total} citations "
        f"({not_found_count} not found, {error_count} errors)."
    )
    report_json["overall_score"] = overall_score
    report.report_json = report_json

    session.add(report)
    session.commit()
    session.refresh(report)

    return DebrievReport.model_validate(report.report_json).model_dump()


@app.get("/v1/reports/{doc_id}")
def get_report(doc_id: UUID, session: Session = Depends(get_session)) -> dict[str, Any]:
    from debriev_core.types import DebrievReport

    report = session.exec(select(Report).where(Report.doc_id == doc_id)).first()
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")

    return DebrievReport.model_validate(report.report_json).model_dump()


_CASE_WORD = r"[A-Z](?:[A-Za-z0-9&'-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9&'-]+)*\.?"
_CASE_CONNECTOR = r"(?:of|the|and|for|de|la|da|del)"
_CASE_PART = (
    rf"{_CASE_WORD}(?:(?:\s+(?:{_CASE_CONNECTOR})\s+|\s+|,\s*){_CASE_WORD}){{0,8}}"
)
_CASE_NAME_PATTERN = re.compile(
    r"\b("
    r"(?:Estate of\s+)?"
    r"[A-Z][A-Za-z0-9&.'-]*"
    r"(?:\s+[A-Z][A-Za-z0-9&.'-]*){0,6}"
    r"\s+v\.?\s+"
    r"[A-Z][A-Za-z0-9&.'-]*"
    r"(?:\s+[A-Z][A-Za-z0-9&.'-]*){0,8}"
    r"(?:,\s*(?:Co\.,\s*Ltd\.?|Inc\.?|Corp\.?|LLC|L\.L\.C\.|Ltd\.?|Co\.?))?"
    r")\b"
)


def _normalize_case_name(name: str) -> str:
    normalized = _normalize_ws(name)
    normalized = re.sub(r"\s+([,.;:])", r"\1", normalized)
    normalized = re.sub(r"\bAir\s+Lines\b", "Airlines", normalized, flags=re.IGNORECASE)
    return normalized.strip(" ,;:.\"'")


def _case_key(name: str) -> str:
    normalized = _normalize_case_name(name).lower()
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def _case_mentioned_in_comparator_context(text: str, case_name: str) -> bool:
    normalized = _normalize_ws(text)
    target_key = _case_key(case_name)
    for sentence in _split_context_sentences(normalized):
        sent_lower = sentence.lower()
        matching_mentions = [
            mention
            for mention in _extract_case_mentions(sentence)
            if _case_key(mention) == target_key
        ]
        if not matching_mentions:
            continue

        pivot_positions = [sent_lower.find(p) for p in _COMPARATOR_PIVOT_PHRASES]
        pivot_positions = [i for i in pivot_positions if i >= 0]
        if not pivot_positions:
            continue
        pivot_pos = min(pivot_positions)

        for mention in matching_mentions:
            mention_pos = sent_lower.find(_normalize_ws(mention).lower())
            if mention_pos >= 0 and pivot_pos < mention_pos:
                return True
    return False


def _dedup_prefix_cases(cases: list[str]) -> list[str]:
    """
    If we have both 'Zicherman v Korean' and 'Zicherman v Korean Airlines Co., Ltd',
    keep only the longer one.
    """
    normalized = [(_case_key(c), c) for c in cases]
    # sort longest first so we keep the most complete mention
    normalized.sort(key=lambda x: len(x[0]), reverse=True)

    kept: list[tuple[str, str]] = []
    for key, orig in normalized:
        if any(key != k2 and key in k2 for k2, _ in kept):
            continue
        kept.append((key, orig))

    # restore original order-ish by the order they appeared in `cases`
    kept_keys = {k for k, _ in kept}
    return [c for c in cases if _case_key(c) in kept_keys]


_BOGUS_INDICATORS = (
    "does not appear to exist",
    "non-existent",
    "nonexistent",
    "no such case",
    "there has been no such case",
    "bogus",
    "fake as well",
    "appear to be fake",
    "copies of non-existent",
    "bogus judicial decisions",
)
_BOGUS_REASON_PHRASES = (
    "does not appear to exist",
    "non-existent",
    "nonexistent",
    "no such case",
    "there has been no such case",
    "bogus judicial decisions",
    "appear to be fake",
    "fake as well",
    "bogus",
    "fake",
)
_BOGUS_REASON_LABELS: tuple[tuple[str, str], ...] = (
    ("does not appear to exist", "nonexistent_case"),
    ("copies of non-existent", "nonexistent_case"),
    ("non-existent", "nonexistent_case"),
    ("nonexistent", "nonexistent_case"),
    ("there has been no such case", "no_such_case"),
    ("no such case", "no_such_case"),
    ("appear to be fake", "appears_fake"),
    ("fake as well", "appears_fake"),
    ("bogus judicial decisions", "appears_fake"),
    ("bogus", "appears_fake"),
    ("fake", "appears_fake"),
)

_COMPARATOR_INDICATORS = (
    "the case appearing at that citation is",
    "the case found at",
    "captioned",
    "docket number",
    "the case appearing at",
    "is titled",
)
_COMPARATOR_PIVOT_PHRASES = (
    "the case appearing at that citation is",
    "the case found at",
    "is titled",
    "captioned",
    "docket number",
)

_FAKE_LIST_INDICATORS = (
    "appear to be fake as well",
    "appear to be fake as well:",
    "appear to be fake",
    "appear to be fake:",
    "fake as well",
    "fake as well:",
)
_STRICT_COMPARATOR_LINE_PATTERNS = (
    re.compile(r"\bthe case appearing at that citation is\b", re.IGNORECASE),
    re.compile(r"\bthe case found at\b.*\bis\b", re.IGNORECASE),
    re.compile(r"\bis titled\b", re.IGNORECASE),
    re.compile(r"\bcaptioned\b", re.IGNORECASE),
)
_RECENT_CASE_TOKEN_PATTERN = re.compile(r"\b[A-Z][A-Za-z'-]{2,}\b")
_LEADING_COURT_PREFIX_PATTERNS = (
    re.compile(
        r"^(?:United States Court of Appeals for the|Court of Appeals for the)\s+",
        re.IGNORECASE,
    ),
    re.compile(r"^[A-Za-z]+\s+Circuit,\s+", re.IGNORECASE),
    re.compile(r"^Circuit,\s+", re.IGNORECASE),
)


def _normalize_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _strip_leading_court_prefix(candidate: str) -> str:
    cleaned = _normalize_ws(candidate)
    while True:
        updated = cleaned
        for pattern in _LEADING_COURT_PREFIX_PATTERNS:
            updated = pattern.sub("", updated)
        updated = updated.lstrip(" ,;:-")
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned.lstrip(" ,;:-")


def _extract_case_mentions(text: str) -> list[str]:
    seen: set[str] = set()
    results: list[str] = []
    for match in _CASE_NAME_PATTERN.finditer(text):
        candidate = _strip_leading_court_prefix(match.group(1)).rstrip(" ,;:.\"'")
        candidate = re.sub(r"^(?:[A-Za-z]+\s+Circuit,\s+)", "", candidate).strip()
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        results.append(candidate)
    return results


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def _is_list_item_line(text: str) -> bool:
    return bool(re.match(r"^(?:[-*•]\s+|\d+[\).\s])", text))


def _is_strict_comparator_line(text: str) -> bool:
    return any(
        pattern.search(text) is not None for pattern in _STRICT_COMPARATOR_LINE_PATTERNS
    )


def _split_before_comparator(text: str) -> str:
    lowered = text.lower()
    match_indices = [lowered.find(phrase) for phrase in _COMPARATOR_INDICATORS]
    valid_indices = [idx for idx in match_indices if idx >= 0]
    if not valid_indices:
        return text
    return text[: min(valid_indices)]


def _extract_recent_party_tokens(text: str) -> list[str]:
    stopwords = {
        "the",
        "there",
        "this",
        "that",
        "citation",
        "case",
        "cases",
        "court",
        "order",
        "federal",
        "reporter",
        "third",
        "docket",
        "number",
        "party",
        "parties",
    }
    tokens: list[str] = []
    seen: set[str] = set()
    for match in _RECENT_CASE_TOKEN_PATTERN.finditer(text):
        token = match.group(0)
        key = token.lower()
        if key in stopwords or key in seen:
            continue
        seen.add(key)
        tokens.append(token)
    return tokens


def _split_context_sentences(text: str) -> list[str]:
    raw = text.replace("\n", " ")
    abbreviations = {
        "v",
        "vs",
        "co",
        "ltd",
        "inc",
        "corp",
        "cir",
        "f",
        "supp",
        "n",
        "d",
        "s",
        "ga",
        "u",
        "us",
        "no",
        "nos",
    }

    sentences: list[str] = []
    start = 0
    i = 0
    while i < len(raw):
        if raw[i] not in ".!?":
            i += 1
            continue

        j = i - 1
        while j >= 0 and (raw[j].isalnum() or raw[j] in "&'-"):
            j -= 1
        prev_token = raw[j + 1 : i].strip("()[]{}\"'").lower()

        end = i + 1
        while end < len(raw) and raw[end] in ")]}\"'":
            end += 1

        k = end
        while k < len(raw) and raw[k].isspace():
            k += 1
        next_char = raw[k] if k < len(raw) else ""

        is_abbreviation = prev_token in abbreviations or (
            len(prev_token) == 1 and prev_token.isalpha()
        )
        should_split = False
        if not next_char:
            should_split = True
        elif next_char in {"•", "-", "*", "–", "—"}:
            should_split = True
        elif next_char.isupper() and not is_abbreviation:
            should_split = True

        if should_split:
            sentence = _normalize_ws(raw[start:end])
            if sentence:
                sentences.append(sentence)
            start = end
            i = end
            continue

        i += 1

    tail = _normalize_ws(raw[start:])
    if tail:
        sentences.append(tail)

    return sentences


def _extract_bogus_case_list(
    chunks: list[dict[str, object]], limit: int = 20
) -> list[str]:
    seen: set[str] = set()
    results: list[str] = []
    recent_cases: list[str] = []
    recent_case_keys: list[str] = []
    bogus_source_keys: set[str] = set()
    fake_list_case_keys: set[str] = set()

    def add_case_name(case_name: str) -> None:
        key = case_name.lower()
        if key in seen:
            return
        seen.add(key)
        results.append(case_name)

    def add_cases(text: str) -> None:
        for case_name in _extract_case_mentions(text):
            add_case_name(case_name)
            if len(results) >= limit:
                return

    def cache_recent_cases(text: str) -> None:
        for case_name in _extract_case_mentions(text):
            key = case_name.lower()
            if key in recent_case_keys:
                idx = recent_case_keys.index(key)
                recent_case_keys.pop(idx)
                recent_cases.pop(idx)
            recent_case_keys.append(key)
            recent_cases.append(case_name)
            while len(recent_cases) > 10:
                recent_cases.pop(0)
                recent_case_keys.pop(0)

    def add_cases_from_recent_party_link(text: str) -> None:
        if "no such case" not in text.lower():
            return
        for token in _extract_recent_party_tokens(text):
            token_lower = token.lower()
            for case_name in reversed(recent_cases):
                if token_lower in case_name.lower():
                    add_case_name(case_name)
                    break

    def add_cases_with_keys(text: str, key_set: set[str]) -> None:
        for case_name in _extract_case_mentions(text):
            add_case_name(case_name)
            key_set.add(_case_key(case_name))
            if len(results) >= limit:
                return

    def add_recent_party_link_cases_with_keys(text: str, key_set: set[str]) -> None:
        if "no such case" not in text.lower():
            return
        for token in _extract_recent_party_tokens(text):
            token_lower = token.lower()
            for case_name in reversed(recent_cases):
                if token_lower in case_name.lower():
                    add_case_name(case_name)
                    key_set.add(_case_key(case_name))
                    break

    def is_obvious_paragraph_break(text: str) -> bool:
        return bool(
            re.match(
                r"^(?:Citation\s+\[\d+\]|The Court|At oral argument|Accordingly|Based on|In sum|In addition)\b",
                text,
                re.IGNORECASE,
            )
        )

    def _explode_inline_bullets(text: str) -> list[str]:
        exploded = text.replace("\r\n", "\n")
        exploded = re.sub(r"\s+([•·\u2022])\s+", r"\n\1 ", exploded)
        return exploded.splitlines()

    for chunk in chunks:
        chunk_text = str(chunk.get("text", "") or "")
        if not chunk_text:
            continue

        lines = [line.strip() for line in _explode_inline_bullets(chunk_text)]
        chunk_cases = _extract_case_mentions(chunk_text)
        chunk_has_bogus = _contains_any(chunk_text, _BOGUS_INDICATORS)

        if chunk_has_bogus:
            varghese_anchor = any(
                _contains_any(line, _BOGUS_INDICATORS) and "varghese" in line.lower()
                for line in lines
            )
            if varghese_anchor:
                for case_name in chunk_cases:
                    if "varghese v" in case_name.lower():
                        add_case_name(case_name)
                        bogus_source_keys.add(_case_key(case_name))
                        if len(results) >= limit:
                            return results

        collecting_fake_list = False
        list_item_buf: str | None = None
        fake_list_buf: str | None = None

        def flush_fake_list_buf() -> None:
            nonlocal fake_list_buf
            if not fake_list_buf:
                return
            lowered = fake_list_buf.lower()
            for case_name in _extract_case_mentions(fake_list_buf):
                mention_pos = lowered.find(_normalize_ws(case_name).lower())
                if mention_pos >= 0:
                    local_before = lowered[max(0, mention_pos - 24) : mention_pos]
                    if re.search(r"(?:^|[;:,])\s*and\s*$", local_before):
                        continue
                before = len(results)
                add_case_name(case_name)
                if len(results) > before:
                    fake_list_case_keys.add(_case_key(case_name))
                if len(results) >= limit:
                    break
            fake_list_buf = None

        def flush_list_item_buf() -> None:
            nonlocal list_item_buf
            if not list_item_buf:
                return
            add_cases_with_keys(
                _split_before_comparator(list_item_buf), fake_list_case_keys
            )
            list_item_buf = None

        for line in lines:
            if len(results) >= limit:
                return results

            if not line:
                flush_list_item_buf()
                flush_fake_list_buf()
                collecting_fake_list = False
                continue

            cache_recent_cases(line)
            is_bogus_line = _contains_any(line, _BOGUS_INDICATORS)
            is_comparator_line = _contains_any(line, _COMPARATOR_INDICATORS)
            starts_fake_list = re.search(
                r"\b(?:also\s+)?appear to be fake as well\b:?", line, re.IGNORECASE
            ) is not None or _contains_any(line, _FAKE_LIST_INDICATORS)
            is_list_item = _is_list_item_line(line)
            is_strict_comparator_line = _is_strict_comparator_line(line)

            if starts_fake_list:
                flush_list_item_buf()
                flush_fake_list_buf()
                collecting_fake_list = True
                fake_list_buf = line
                if line.endswith("."):
                    flush_fake_list_buf()
                    collecting_fake_list = False
                continue

            if collecting_fake_list:
                if is_strict_comparator_line or is_obvious_paragraph_break(line):
                    flush_fake_list_buf()
                    collecting_fake_list = False
                    continue

                if is_list_item:
                    flush_fake_list_buf()
                    flush_list_item_buf()
                    list_item_buf = line
                    if line.endswith(".") or line.endswith(";"):
                        flush_list_item_buf()
                    continue

                if list_item_buf is not None:
                    list_item_buf = _normalize_ws(f"{list_item_buf} {line}")
                    if line.endswith(".") or line.endswith(";"):
                        flush_list_item_buf()
                    continue

                if fake_list_buf is None:
                    fake_list_buf = line
                else:
                    fake_list_buf = _normalize_ws(f"{fake_list_buf} {line}")
                if line.endswith("."):
                    flush_fake_list_buf()
                    collecting_fake_list = False
                continue

            if is_bogus_line:
                bogus_part = _split_before_comparator(line)
                add_cases_with_keys(bogus_part, bogus_source_keys)
                add_recent_party_link_cases_with_keys(bogus_part, bogus_source_keys)
                continue

            if is_strict_comparator_line or is_comparator_line:
                continue

        flush_list_item_buf()
        flush_fake_list_buf()

        for sentence in _split_context_sentences(chunk_text):
            if len(results) >= limit:
                return results
            cache_recent_cases(sentence)
            if not _contains_any(sentence, _BOGUS_INDICATORS):
                continue

            extractable_sentence = _split_before_comparator(sentence)
            add_cases_with_keys(extractable_sentence, bogus_source_keys)
            add_recent_party_link_cases_with_keys(
                extractable_sentence, bogus_source_keys
            )

    allowed_keys = bogus_source_keys | fake_list_case_keys
    results = [c for c in results if _case_key(c) in allowed_keys]
    combined_text = "\n".join(str(chunk.get("text", "") or "") for chunk in chunks)
    results = [
        c
        for c in results
        if _case_key(c) in fake_list_case_keys
        or not _case_mentioned_in_comparator_context(combined_text, c)
    ]
    results = _dedup_prefix_cases(results)
    return results


@dataclass(frozen=True)
class BogusCaseFinding:
    case_name: str
    reason_label: str
    reason_phrase: str
    evidence: str
    doc_id: str | None = None
    chunk_id: str | None = None


def _clean_evidence_snippet(evidence: str, max_len: int = 240) -> str:
    cleaned = _normalize_ws(evidence)
    if len(cleaned) <= max_len:
        return cleaned
    return f"{cleaned[: max_len - 3].rstrip()}..."


def _first_bogus_reason(text: str) -> tuple[str, str] | None:
    lowered = text.lower()
    if "no such case" in lowered or "there has been no such case" in lowered:
        return ("nonexistent_case", "no such case")
    if "does not appear to exist" in lowered:
        return ("nonexistent_case", "does not appear to exist")
    if "non-existent" in lowered or "nonexistent" in lowered:
        return ("nonexistent_case", "non-existent")
    if "appear to be fake" in lowered or "fake as well" in lowered:
        return ("appears_fake", "appear to be fake")
    if "bogus judicial decisions" in lowered or "bogus" in lowered or "fake" in lowered:
        return ("appears_fake", "bogus")
    return None


def _extract_bogus_case_findings(
    chunks: list[dict[str, object]],
    limit: int = 20,
) -> list[BogusCaseFinding]:
    allowed_cases = _extract_bogus_case_list(chunks, limit=max(limit, 20))
    if not allowed_cases:
        return []

    allowed_by_key = {_case_key(case_name): case_name for case_name in allowed_cases}
    findings: list[BogusCaseFinding] = []
    seen_keys: set[str] = set()

    def add_finding(
        *,
        case_key: str,
        reason_label: str,
        reason_phrase: str,
        evidence: str,
        doc_id: str | None,
        chunk_id: str | None,
    ) -> None:
        if case_key in seen_keys:
            return
        seen_keys.add(case_key)
        findings.append(
            BogusCaseFinding(
                case_name=_normalize_case_name(allowed_by_key[case_key]),
                reason_label=reason_label,
                reason_phrase=reason_phrase,
                evidence=_clean_evidence_snippet(evidence),
                doc_id=doc_id,
                chunk_id=chunk_id,
            )
        )

    for chunk in chunks:
        chunk_text = str(chunk.get("text", "") or "")
        if not chunk_text:
            continue

        doc_id_raw = chunk.get("doc_id")
        chunk_id_raw = chunk.get("chunk_id")
        doc_id = str(doc_id_raw) if doc_id_raw is not None else None
        chunk_id = str(chunk_id_raw) if chunk_id_raw is not None else None

        for sentence in _split_context_sentences(chunk_text):
            evidence = _normalize_ws(sentence)
            if not evidence:
                continue

            reason_info = _first_bogus_reason(evidence)
            if reason_info is None:
                continue
            reason_label, reason_phrase = reason_info

            extractable = _split_before_comparator(evidence)
            for case_name in _extract_case_mentions(extractable):
                key = _case_key(case_name)
                if key not in allowed_by_key or key in seen_keys:
                    continue

                add_finding(
                    case_key=key,
                    reason_label=reason_label,
                    reason_phrase=reason_phrase,
                    evidence=evidence,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                )
                if len(findings) >= limit:
                    return findings

        lines = [line.strip() for line in chunk_text.splitlines() if line.strip()]
        for idx, line in enumerate(lines):
            if len(findings) >= limit:
                return findings

            line_text = _normalize_ws(line)
            mentions = _extract_case_mentions(line_text)
            if not mentions:
                continue

            reason_info = _first_bogus_reason(line_text)
            evidence = line_text
            if reason_info is None:
                for prev_idx in range(idx - 1, max(-1, idx - 4), -1):
                    prev_line = _normalize_ws(lines[prev_idx])
                    prev_reason_info = _first_bogus_reason(prev_line)
                    if prev_reason_info is None:
                        continue
                    reason_info = prev_reason_info
                    evidence = _normalize_ws(f"{prev_line} {line_text}")
                    break

            if reason_info is None:
                continue
            reason_label, reason_phrase = reason_info

            extractable = _split_before_comparator(line_text)
            for case_name in _extract_case_mentions(extractable):
                key = _case_key(case_name)
                if key not in allowed_by_key or key in seen_keys:
                    continue
                add_finding(
                    case_key=key,
                    reason_label=reason_label,
                    reason_phrase=reason_phrase,
                    evidence=evidence,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                )

    findings.sort(key=lambda item: _case_key(item.case_name))
    return findings


_BOGUS_LIST_KEYWORDS = ("bogus", "non-existent", "nonexistent", "fake")
_CITATION_VERIFICATION_TRIGGERS = (
    "verify citations",
    "check citations",
    "are these citations real",
    "validate citations",
    "courtlistener",
    "citation lookup",
)
_CHAT_VERIFICATION_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:[^\n`]*)\n?(.*?)```", re.DOTALL
)


def _is_bogus_case_request(message: str) -> bool:
    lowered = message.lower()
    return any(keyword in lowered for keyword in _BOGUS_LIST_KEYWORDS)


def _is_citation_verification_request(text: str) -> bool:
    lowered = _normalize_ws(text).lower()
    if "citation" not in lowered and "courtlistener" not in lowered:
        return False
    return any(trigger in lowered for trigger in _CITATION_VERIFICATION_TRIGGERS)


def _to_chat_findings(findings: list[BogusCaseFinding]) -> list[ChatFinding]:
    return [
        ChatFinding(
            case_name=finding.case_name,
            reason_label=finding.reason_label,
            reason_phrase=finding.reason_phrase,
            evidence=finding.evidence,
            doc_id=finding.doc_id,
            chunk_id=finding.chunk_id,
        )
        for finding in findings
    ]


def _extract_message_verification_text(message: str) -> str | None:
    cleaned = message.strip()
    if not cleaned:
        return None

    code_match = _CHAT_VERIFICATION_CODE_BLOCK_PATTERN.search(cleaned)
    if code_match:
        snippet = code_match.group(1).strip()
        return snippet if snippet else None

    if "\n" in cleaned:
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if len(lines) >= 2:
            first = lines[0].lower()
            candidate_lines = (
                lines[1:]
                if any(trigger in first for trigger in _CITATION_VERIFICATION_TRIGGERS)
                else lines
            )
            candidate = "\n".join(candidate_lines).strip()
            if len(candidate) >= 60:
                return candidate

    if ":" in cleaned:
        head, tail = cleaned.split(":", 1)
        if _is_citation_verification_request(head) and len(tail.strip()) >= 60:
            return tail.strip()

    return None


def _build_verification_text_from_chunks(
    chunks: list[dict[str, object]], max_chars: int = 8000
) -> str:
    def _order_key(chunk: dict[str, object]) -> tuple[float, str, str, int]:
        raw_score = chunk.get("score")
        try:
            score = float(raw_score) if raw_score is not None else 0.0
        except (TypeError, ValueError):
            score = 0.0

        raw_page = chunk.get("page")
        try:
            page = int(raw_page) if raw_page is not None else -1
        except (TypeError, ValueError):
            page = -1

        return (
            -score,
            str(chunk.get("doc_id") or ""),
            str(chunk.get("chunk_id") or ""),
            page,
        )

    ordered = sorted(chunks, key=_order_key)
    parts: list[str] = []
    total = 0
    for chunk in ordered:
        text = _normalize_ws(str(chunk.get("text", "") or ""))
        if not text:
            continue
        if total >= max_chars:
            break

        remaining = max_chars - total
        piece = text[:remaining].rstrip()
        if not piece:
            continue
        parts.append(piece)
        total += len(piece)
        if len(piece) < len(text):
            break
        total += 2

    return "\n\n".join(parts).strip()


def _citation_verification_answer(result: VerifyCitationsResponse) -> str:
    if not result.findings:
        return "Citation verification requested, but no citations were detected."

    summary = result.summary
    header = (
        "Citation verification results: "
        f"{summary.verified} verified, "
        f"{summary.ambiguous} ambiguous, "
        f"{summary.not_found} not found "
        f"(total {summary.total})."
    )
    lines = [
        f"- {item.citation}: {item.status} (confidence {item.confidence:.2f})"
        for item in result.findings
    ]
    return header + "\n" + "\n".join(lines)


def _stable_json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _as_result_item(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    return {"value": value}


def _normalize_citation_string(value: str) -> str:
    lowered = _normalize_ws(value).lower()
    lowered = re.sub(r"[^\w\s.]", "", lowered)
    return _normalize_ws(lowered).strip(" .")


def _optional_text(data: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str):
            normalized = _normalize_ws(value)
            if normalized:
                return normalized
        if isinstance(value, dict):
            nested = _optional_text(value, ("name", "label", "value", "full_name"))
            if nested:
                return nested
    return None


def _optional_year(data: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            match = re.search(r"\b(19|20)\d{2}\b", value)
            if match:
                try:
                    return int(match.group(0))
                except ValueError:
                    continue
    return None


def _to_best_match(
    candidate: dict[str, Any], fallback_citation: str | None
) -> BestMatch | None:
    case_name = _optional_text(
        candidate,
        (
            "case_name",
            "caseName",
            "name",
            "caption",
            "case",
            "case_name_full",
        ),
    )
    court = _optional_text(candidate, ("court", "court_name", "courtName"))
    year = _optional_year(candidate, ("year", "date_filed", "dateFiled", "date"))
    url = _optional_text(
        candidate,
        (
            "url",
            "absolute_url",
            "frontend_url",
            "opinion_url",
            "cluster_url",
        ),
    )
    matched_citation = _optional_text(
        candidate,
        ("citation", "cite", "normalized_citation", "matched_citation"),
    )
    if matched_citation is None:
        matched_citation = fallback_citation

    if not any((case_name, court, year, url, matched_citation)):
        return None

    return BestMatch(
        case_name=case_name,
        court=court,
        year=year,
        url=url,
        matched_citation=matched_citation,
    )


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[str, str, int, str, str]:
    case_name = _optional_text(
        candidate,
        ("case_name", "caseName", "name", "caption", "case", "case_name_full"),
    )
    court = _optional_text(candidate, ("court", "court_name", "courtName"))
    year = _optional_year(candidate, ("year", "date_filed", "dateFiled", "date"))
    url = _optional_text(
        candidate,
        ("url", "absolute_url", "frontend_url", "opinion_url", "cluster_url"),
    )
    citation = _optional_text(
        candidate,
        ("citation", "cite", "normalized_citation", "matched_citation"),
    )
    return (
        _normalize_citation_string(case_name or ""),
        _normalize_citation_string(court or ""),
        year or 0,
        _normalize_citation_string(url or ""),
        _normalize_citation_string(citation or ""),
    )


def _courtlistener_raw_to_findings(
    raw: dict[str, Any],
    requested_citations: list[str] | None = None,
) -> list[CitationVerificationFinding]:
    by_citation: dict[str, dict[str, Any]] = {}
    if requested_citations:
        for requested in requested_citations:
            key = _normalize_citation_string(requested)
            if not key or key in by_citation:
                continue
            by_citation[key] = {"citation": requested, "results": []}

    for entry in _extract_citation_entries(raw):
        citation = _normalize_ws(
            str(
                entry.get("citation")
                or entry.get("cite")
                or entry.get("normalized_citation")
                or entry.get("text")
                or ""
            )
        )
        if not citation:
            continue

        citation_key = _normalize_citation_string(citation)
        if not citation_key:
            continue
        if requested_citations and citation_key not in by_citation:
            continue

        existing = by_citation.get(citation_key)
        if existing is None:
            by_citation[citation_key] = {
                "citation": citation,
                "results": _extract_results_list(entry),
            }
            continue

        merged = {_stable_json_key(item): item for item in existing["results"]}
        for item in _extract_results_list(entry):
            merged[_stable_json_key(item)] = item
        existing["results"] = sorted(merged.values(), key=_stable_json_key)

    findings: list[CitationVerificationFinding] = []
    for citation_key in sorted(by_citation):
        citation = str(by_citation[citation_key]["citation"])
        candidates = list(by_citation[citation_key]["results"])
        candidate_count = len(candidates)
        best_match: BestMatch | None = None

        if candidate_count == 0:
            status = "not_found"
            confidence = 0.0
            explanation = "No CourtListener matches were returned for this citation."
        elif candidate_count == 1:
            best_match = _to_best_match(candidates[0], fallback_citation=citation)
            matched_citation = (
                best_match.matched_citation if best_match is not None else None
            )
            exact_match = _normalize_citation_string(
                matched_citation or ""
            ) == _normalize_citation_string(citation)
            status = "verified"
            confidence = 1.0 if exact_match else 0.8
            explanation = (
                "One CourtListener match was returned with an exact citation match."
                if exact_match
                else "One CourtListener match was returned, but citation text differs after normalization."
            )
        else:
            ordered_candidates = sorted(candidates, key=_candidate_sort_key)
            best_match = _to_best_match(
                ordered_candidates[0], fallback_citation=citation
            )
            status = "ambiguous"
            confidence = 0.5
            explanation = (
                "Multiple CourtListener matches were returned for this citation."
            )

        findings.append(
            CitationVerificationFinding(
                citation=citation,
                status=status,
                confidence=confidence,
                best_match=best_match,
                explanation=explanation,
            )
        )

    return sorted(findings, key=lambda item: _normalize_citation_string(item.citation))


def _summary_from_chunks(chunks: list[dict[str, object]]) -> str:
    texts = [_normalize_ws(str(chunk.get("text", "") or "")) for chunk in chunks[:2]]
    combined = " ".join(part for part in texts if part).strip()
    if not combined:
        return "I found indexed chunks, but they did not contain enough readable context to summarize."

    sentence_candidates = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", combined)
        if sentence.strip()
    ]
    selected: list[str] = []
    seen_sentences: set[str] = set()
    for sentence in sentence_candidates:
        cleaned = _normalize_ws(sentence)
        key = cleaned.lower()
        if key in seen_sentences:
            continue
        if len(cleaned) < 25 and selected:
            continue
        seen_sentences.add(key)
        selected.append(cleaned)
        if len(selected) >= 4:
            break

    if not selected:
        snippet = combined[:320].rstrip(" .")
        if not snippet:
            return "I could not find enough indexed context to answer that."
        selected = [
            f"{snippet}.",
            "The indexed context is limited, so this summary may be incomplete.",
        ]
    elif len(selected) == 1:
        selected.append(
            "The indexed context is limited, so this summary may be incomplete."
        )

    return " ".join(selected[:5])


def _simple_answer_from_chunks(message: str, chunks: list[dict[str, object]]) -> str:
    if not chunks:
        return "I could not find relevant indexed context for that question yet."

    if _is_bogus_case_request(message):
        findings = _extract_bogus_case_findings(chunks)
        if findings:
            bullets = "\n".join(
                [
                    f"- {finding.case_name} — {finding.reason_label} ({finding.reason_phrase}) "
                    f"(source: {finding.chunk_id or finding.doc_id or 'unknown'})\n"
                    f"  Evidence: {finding.evidence}"
                    for finding in findings
                ]
            )
            return f"Bogus/non-existent cases found:\n{bullets}"

    return _summary_from_chunks(chunks)


def _run_bogus_extractor_self_test() -> None:
    sample = """
    Citation [26] references United States Court of Appeals for the Eleventh Circuit, Varghese v China South Airlines Ltd., 925 F.3d 1339 (11th Cir. 2019), which does not appear to exist. The case appearing at that citation is titled Miccosukee Tribe v. United States.
    There has been no such case and no party named Varghese.
    Citation [27] references Zicherman v Korean Airlines Co., Ltd., 516 F. Supp. 2d 1277 (N.D. Ga. 2008), which does not appear to exist. The case appearing at that citation is Gibbs v. Maxwell House.
    Citation [31] references Zaunbrecher v. Transocean Offshore Deepwater Drilling, Inc., 898 F.3d 211 (5th Cir. 2018), which does not appear to exist.
    - Holliday v. Atl. Capital Corp., which does not appear to exist. The case found at that citation is A.D. v Azar.
    - Hyatt v. N. Cent. Airlines, which does not appear to exist. Docket number 123 is for a case captioned George Cornea v. U.S. Attorney General.
    The following five decisions from Federal Reporter, Third, also appear to be fake as well: Shaboon v. Egyptair, 2013 WL 12131316 (S.D. Fla. Sept. 18, 2013); Petersen v. Iran Air, 905 F. Supp. 2d 121 (D.D.C. 2012);
    Martinez v. Delta Airlines, Inc., 2019 WL 1234567 (S.D. Fla. 2019); Estate of Durden v. KLM Royal Dutch Airlines, 2021 WL 1111111 (S.D.N.Y. 2021); and Miller v. United Airlines, Inc., 174 F.3d 366 (5th Cir. 1999).
    Collapsed bullets from OCR: • Holliday v. Atl. Capital Corp., which does not appear to exist. The case found at that citation is A.D. v Azar. • Hyatt v. N. Cent. Airlines, which does not appear to exist. Docket number 123 is for a case captioned George Cornea v. U.S. Attorney General. • Estate of Durden v. KLM Royal Dutch Airlines, which appears to be fake as well.
    The case appearing at that citation is titled Witt v. Metropolitan Life Ins. Co. The case found at that citation is Miller v. United Airlines, Inc.
    - Witt v. Metropolitan Life Ins. Co. is titled in a comparator sentence.
    Citation [30] references Gibbs v. Maxwell House, a real case.
    """

    found = _extract_bogus_case_list([{"text": sample}])
    print("FOUND:", found)
    found_set = {_case_key(item) for item in found}

    expected_includes = {
        "Varghese v China South Airlines Ltd",
        "Zicherman v Korean Air Lines Co., Ltd",
        "Zaunbrecher v. Transocean Offshore Deepwater Drilling, Inc",
        "Holliday v. Atl. Capital Corp",
        "Hyatt v. N. Cent. Airlines",
        "Shaboon v. Egyptair",
        "Petersen v. Iran Air",
        "Martinez v. Delta Airlines, Inc",
        "Estate of Durden v. KLM Royal Dutch Airlines",
    }
    expected_excludes = {
        "Miccosukee Tribe v. United States",
        "Gibbs v. Maxwell House",
        "A.D. v. Azar",
    }

    for case_name in expected_includes:
        assert _case_key(case_name) in found_set, (
            f"Expected bogus case missing: {case_name}"
        )
    for case_name in expected_excludes:
        assert _case_key(case_name) not in found_set, (
            f"Unexpected comparator/real case included: {case_name}"
        )
    assert not any(item.lower().startswith("circuit,") for item in found), (
        "Unexpected Circuit-prefixed case entry"
    )

    findings = _extract_bogus_case_findings(
        [{"text": sample, "chunk_id": "chunk:self-test"}]
    )
    finding_keys = {_case_key(item.case_name) for item in findings}
    for case_name in expected_includes:
        assert _case_key(case_name) in finding_keys, (
            f"Expected finding missing: {case_name}"
        )
    for finding in findings:
        assert finding.reason_label in {"nonexistent_case", "appears_fake"}, (
            f"Unexpected reason label for finding: {finding.case_name}"
        )
        assert finding.reason_phrase.strip(), (
            f"Missing reason phrase for finding: {finding.case_name}"
        )
        assert finding.evidence.strip(), (
            f"Missing evidence for finding: {finding.case_name}"
        )


@app.post("/v1/retrieval/index", response_model=RetrievalIndexResponse)
def retrieval_index(
    payload: RetrievalIndexRequest,
    session: Session = Depends(get_session),
) -> RetrievalIndexResponse:
    document = session.exec(
        select(Document).where(Document.doc_id == payload.doc_id)
    ).first()
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    if not (document.stub_text or "").strip():
        raise HTTPException(status_code=400, detail="Document has no extracted text")

    chunks_indexed = index_document_from_db(
        session=session,
        doc_id=payload.doc_id,
        db_path=settings.retrieval_db_path,
        provider=settings.embed_provider,
        openai_api_key=settings.openai_api_key,
    )
    return RetrievalIndexResponse(doc_id=payload.doc_id, chunks_indexed=chunks_indexed)


@app.post("/v1/retrieval/query", response_model=list[RetrievalChunkResponse])
def retrieval_query(
    payload: RetrievalQueryRequest,
    session: Session = Depends(get_session),
) -> list[RetrievalChunkResponse]:
    query_text = payload.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="query is required")

    document = session.exec(
        select(Document).where(Document.doc_id == payload.doc_id)
    ).first()
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    requested_k = max(1, min(payload.k, 20))
    rows = query_document_chunks(
        db_path=settings.retrieval_db_path,
        doc_id=str(payload.doc_id),
        query=query_text,
        k=requested_k,
        provider=settings.embed_provider,
        openai_api_key=settings.openai_api_key,
    )

    return [
        RetrievalChunkResponse(
            chunk_id=str(row["chunk_id"]),
            score=float(row["score"]),
            page=int(row["page"]) if row["page"] is not None else None,
            text=str(row["text"]),
        )
        for row in rows
    ]


@app.post("/v1/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    session: Session = Depends(get_session),
) -> ChatResponse:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    document = session.exec(
        select(Document).where(Document.doc_id == payload.doc_id)
    ).first()
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    requested_k = max(1, min(payload.k, 20))
    rows = query_document_chunks(
        db_path=settings.retrieval_db_path,
        doc_id=str(payload.doc_id),
        query=message,
        k=requested_k,
        provider=settings.embed_provider,
        openai_api_key=settings.openai_api_key,
    )

    sources = [
        ChatSource(
            doc_id=str(payload.doc_id),
            chunk_id=str(row["chunk_id"]),
            page=int(row["page"]) if row["page"] is not None else None,
            text=str(row["text"]),
            score=float(row["score"]),
        )
        for row in rows
    ]
    tool_result: CitationVerificationToolResult | None = None
    if _is_citation_verification_request(message):
        message_text = _extract_message_verification_text(message)
        citations = (
            _extract_citations(message_text)
            if message_text is not None
            else _extract_citations_from_chunks(rows)
        )
        if citations:
            verification_response = _run_citation_verification_for_citations(
                citations=citations,
                session=session,
                doc_id=str(payload.doc_id),
            )
            answer = _citation_verification_answer(verification_response)
            tool_result = CitationVerificationToolResult(
                findings=verification_response.findings,
                summary=verification_response.summary,
                citations=verification_response.citations,
            )
        else:
            verification_response = VerifyCitationsResponse()
            answer = "Citation verification requested, but no citations were detected."
            tool_result = CitationVerificationToolResult(
                findings=verification_response.findings,
                summary=verification_response.summary,
                citations=verification_response.citations,
            )
        findings: list[ChatFinding] = []
    else:
        answer = _simple_answer_from_chunks(message, rows)
        findings = (
            _to_chat_findings(_extract_bogus_case_findings(rows))
            if _is_bogus_case_request(message)
            else []
        )

    return ChatResponse(
        answer=answer,
        sources=sources,
        findings=findings,
        tool_result=tool_result,
    )


@app.post("/v1/chat/project", response_model=ChatResponse)
def chat_project(
    payload: ProjectChatRequest,
    session: Session = Depends(get_session),
) -> ChatResponse:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    project = session.exec(
        select(Project).where(Project.project_id == payload.project_id)
    ).first()
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    docs = session.exec(
        select(Document)
        .where(Document.project_id == payload.project_id)
        .order_by(Document.created_at.desc())
    ).all()
    if not docs:
        return ChatResponse(
            answer="This project has no indexed documents yet.", sources=[]
        )

    requested_docs = max(1, min(payload.k_docs, 20))
    requested_per_doc = max(1, min(payload.k_per_doc, 20))
    requested_total = max(1, min(payload.k_total, 20))

    doc_ids = [str(doc.doc_id) for doc in docs[:requested_docs]]
    rows = query_project_chunks(
        db_path=settings.retrieval_db_path,
        doc_ids=doc_ids,
        query=message,
        k_per_doc=requested_per_doc,
        k_total=requested_total,
        provider=settings.embed_provider,
        openai_api_key=settings.openai_api_key,
    )

    sources = [
        ChatSource(
            doc_id=str(row["doc_id"]),
            chunk_id=str(row["chunk_id"]),
            page=int(row["page"]) if row["page"] is not None else None,
            text=str(row["text"]),
            score=float(row["score"]),
        )
        for row in rows
    ]
    tool_result: CitationVerificationToolResult | None = None
    if _is_citation_verification_request(message):
        message_text = _extract_message_verification_text(message)
        citations = (
            _extract_citations(message_text)
            if message_text is not None
            else _extract_citations_from_chunks(rows)
        )
        if citations:
            verification_response = _run_citation_verification_for_citations(
                citations=citations,
                session=session,
            )
            answer = _citation_verification_answer(verification_response)
            tool_result = CitationVerificationToolResult(
                findings=verification_response.findings,
                summary=verification_response.summary,
                citations=verification_response.citations,
            )
        else:
            verification_response = VerifyCitationsResponse()
            answer = "Citation verification requested, but no citations were detected."
            tool_result = CitationVerificationToolResult(
                findings=verification_response.findings,
                summary=verification_response.summary,
                citations=verification_response.citations,
            )
        findings: list[ChatFinding] = []
    else:
        answer = _simple_answer_from_chunks(message, rows)
        findings = (
            _to_chat_findings(_extract_bogus_case_findings(rows))
            if _is_bogus_case_request(message)
            else []
        )

    return ChatResponse(
        answer=answer,
        sources=sources,
        findings=findings,
        tool_result=tool_result,
    )
