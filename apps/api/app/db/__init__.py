from __future__ import annotations

from datetime import UTC, datetime
import json
from typing import Any

try:
    from sqlmodel import Session, select
except (
    ModuleNotFoundError
):  # pragma: no cover - allows extractor self-tests without db deps
    Session = Any  # type: ignore[assignment]

    def select(*args: Any, **kwargs: Any) -> Any:
        raise ModuleNotFoundError("sqlmodel is required for database-backed helpers")


from .models import (
    Citation,
    CitationVerification,
    Document,
    ExtractedCitations,
    Project,
    Report,
    VerificationJob,
    VerificationResult,
)
from .session import get_session, init_db


def store_verification_result(
    session: Session,
    *,
    doc_id: str,
    input_hash: str,
    citations_hash: str,
    citations: list[str],
    findings: list[dict[str, Any]],
    summary: dict[str, Any],
) -> VerificationResult:
    now = datetime.now(UTC)
    citations_json = json.dumps(citations, sort_keys=True)
    findings_json = json.dumps(findings, sort_keys=True)
    summary_json = json.dumps(summary, sort_keys=True)

    created = VerificationResult(
        doc_id=doc_id,
        input_hash=input_hash,
        citations_hash=citations_hash,
        citations_json=citations_json,
        findings_json=findings_json,
        summary_json=summary_json,
        created_at=now,
        updated_at=now,
    )
    session.add(created)
    session.commit()
    session.refresh(created)
    return created


def get_latest_verification_result(
    session: Session, *, doc_id: str
) -> VerificationResult | None:
    return session.exec(
        select(VerificationResult)
        .where(VerificationResult.doc_id == doc_id)
        .order_by(VerificationResult.updated_at.desc(), VerificationResult.id.desc())
    ).first()


def list_verification_results(
    session: Session, *, doc_id: str, limit: int = 100
) -> list[VerificationResult]:
    safe_limit = max(1, min(limit, 500))
    return session.exec(
        select(VerificationResult)
        .where(VerificationResult.doc_id == doc_id)
        .order_by(VerificationResult.created_at.desc(), VerificationResult.id.desc())
        .limit(safe_limit)
    ).all()


def store_extracted_citations(
    session: Session,
    *,
    doc_id: str,
    citations: list[str],
    evidence_map: dict[str, str],
    probable_case_name_map: dict[str, str | None],
) -> ExtractedCitations:
    existing = session.exec(
        select(ExtractedCitations).where(ExtractedCitations.doc_id == doc_id)
    ).first()

    now = datetime.now(UTC)
    citations_json = json.dumps(citations, sort_keys=True)
    evidence_json = json.dumps(
        {key: evidence_map[key] for key in sorted(evidence_map)},
        sort_keys=True,
    )
    probable_json = json.dumps(
        {key: probable_case_name_map[key] for key in sorted(probable_case_name_map)},
        sort_keys=True,
    )

    if existing is not None:
        existing.citations_json = citations_json
        existing.evidence_json = evidence_json
        existing.probable_case_name_json = probable_json
        existing.updated_at = now
        session.add(existing)
        session.commit()
        session.refresh(existing)
        return existing

    created = ExtractedCitations(
        doc_id=doc_id,
        citations_json=citations_json,
        evidence_json=evidence_json,
        probable_case_name_json=probable_json,
        created_at=now,
        updated_at=now,
    )
    session.add(created)
    session.commit()
    session.refresh(created)
    return created


def get_latest_extracted_citations(
    session: Session, *, doc_id: str
) -> ExtractedCitations | None:
    return session.exec(
        select(ExtractedCitations)
        .where(ExtractedCitations.doc_id == doc_id)
        .order_by(ExtractedCitations.updated_at.desc(), ExtractedCitations.id.desc())
    ).first()


__all__ = [
    "Project",
    "Document",
    "Report",
    "Citation",
    "CitationVerification",
    "ExtractedCitations",
    "VerificationJob",
    "VerificationResult",
    "store_extracted_citations",
    "get_latest_extracted_citations",
    "store_verification_result",
    "get_latest_verification_result",
    "list_verification_results",
    "get_session",
    "init_db",
]
