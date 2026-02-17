from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class CitationSpan(BaseModel):
    raw: str
    normalized: str | None = None
    start: int | None = None
    end: int | None = None
    context_text: str | None = None


class VerificationResult(BaseModel):
    status: str
    matched: dict[str, Any] | None = None
    match_score: float | None = None
    details: dict[str, Any] | None = None


class DebrievReport(BaseModel):
    version: str
    overall_score: int
    summary: str
    citations: list[dict[str, Any]]
    created_at: str | None = None
