from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from debriev_core.ingest.pdf_text import extract_pdf_text
from debriev_core.stubs import extract_citations
from debriev_core.types import DebrievReport
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlmodel import Session, select

from app.db import Citation, Document, Report, get_session, init_db
from app.settings import settings

app = FastAPI(title="Debriev API")


class UploadResponse(BaseModel):
    doc_id: UUID


class DocumentResponse(BaseModel):
    doc_id: UUID
    filename: str
    created_at: datetime


class CitationResponse(BaseModel):
    id: int
    doc_id: UUID
    raw: str
    normalized: str | None
    start: int | None
    end: int | None
    context_text: str | None
    created_at: datetime


@app.on_event("startup")
def on_startup() -> None:
    Path(settings.storage_dir).mkdir(parents=True, exist_ok=True)
    init_db()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/upload", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    if Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only .pdf files are supported")

    document = Document(
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

    return UploadResponse(doc_id=document.doc_id)


@app.get("/v1/documents", response_model=list[DocumentResponse])
def list_documents(session: Session = Depends(get_session)) -> list[DocumentResponse]:
    rows = session.exec(select(Document).order_by(Document.created_at.desc())).all()
    return [
        DocumentResponse(doc_id=row.doc_id, filename=row.filename, created_at=row.created_at)
        for row in rows
    ]


@app.get("/v1/citations/{doc_id}", response_model=list[CitationResponse])
def list_citations(doc_id: UUID, session: Session = Depends(get_session)) -> list[CitationResponse]:
    document = session.exec(select(Document).where(Document.doc_id == doc_id)).first()
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    rows = session.exec(select(Citation).where(Citation.doc_id == doc_id).order_by(Citation.id)).all()
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
def verify_report(doc_id: UUID, session: Session = Depends(get_session)) -> dict[str, Any]:
    report = session.exec(select(Report).where(Report.doc_id == doc_id)).first()
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")

    citation_rows = session.exec(
        select(Citation).where(Citation.doc_id == doc_id).order_by(Citation.id)
    ).all()

    report_json = dict(report.report_json)
    citations_payload = report_json.get("citations")
    if isinstance(citations_payload, list):
        for entry in citations_payload:
            if isinstance(entry, dict):
                entry["verification_status"] = "unverified"

    report_json["summary"] = f"Verified {len(citation_rows)} citation(s) (stub)."
    report.report_json = report_json

    session.add(report)
    session.commit()
    session.refresh(report)

    return DebrievReport.model_validate(report.report_json).model_dump()


@app.get("/v1/reports/{doc_id}")
def get_report(doc_id: UUID, session: Session = Depends(get_session)) -> dict[str, Any]:
    report = session.exec(select(Report).where(Report.doc_id == doc_id)).first()
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")

    return DebrievReport.model_validate(report.report_json).model_dump()
