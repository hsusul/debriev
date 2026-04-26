"""Draft intake workflow for MVP pre-review creation."""

from dataclasses import dataclass
import re
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.enums import DraftMode
from app.repositories.assertions import AssertionRepository
from app.repositories.drafts import DraftRepository
from app.repositories.matters import MatterRepository
from app.schemas.draft import DraftCreate, DraftTextCreate
from app.schemas.matter import MatterCreate
from app.services.claims.extractor import ClaimExtractionService
from app.services.parsing.normalization import normalize_for_match

PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(slots=True)
class DraftIntakeResult:
    draft_id: UUID
    matter_id: UUID
    title: str
    assertion_count: int
    claim_count: int


class DraftIntakeService:
    """Create a minimally reviewable draft from pasted or uploaded text."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.matters = MatterRepository(session)
        self.drafts = DraftRepository(session)
        self.assertions = AssertionRepository(session)
        self.extractor = ClaimExtractionService(session)

    def create_draft_from_text(self, payload: DraftTextCreate) -> DraftIntakeResult:
        cleaned_text = payload.draft_text.strip()
        title = _resolve_title(cleaned_text, payload.title)

        matter = self.matters.create(
            MatterCreate(
                name=title,
                status="ACTIVE",
            )
        )
        draft = self.drafts.create(
            matter.id,
            DraftCreate(
                title=title,
                mode=DraftMode.DRAFT,
            ),
        )

        claim_count = 0
        assertion_count = 0
        for paragraph_index, chunk in enumerate(_split_assertions(cleaned_text), start=1):
            assertion = self.assertions.create(
                draft.id,
                paragraph_index=paragraph_index,
                sentence_index=None,
                raw_text=chunk,
                normalized_text=normalize_for_match(chunk),
            )
            assertion_count += 1
            claim_count += len(self.extractor.extract_from_assertion(assertion.id))

        self.session.flush()
        return DraftIntakeResult(
            draft_id=draft.id,
            matter_id=matter.id,
            title=title,
            assertion_count=assertion_count,
            claim_count=claim_count,
        )


def _split_assertions(raw_text: str) -> list[str]:
    paragraphs = [
        _normalize_assertion_text(chunk)
        for chunk in PARAGRAPH_SPLIT_RE.split(raw_text)
        if chunk.strip()
    ]
    if paragraphs:
        return paragraphs

    normalized = _normalize_assertion_text(raw_text)
    return [normalized] if normalized else []


def _normalize_assertion_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    collapsed = " ".join(lines)
    return WHITESPACE_RE.sub(" ", collapsed).strip()


def _resolve_title(draft_text: str, explicit_title: str | None) -> str:
    if explicit_title is not None and explicit_title.strip():
        return explicit_title.strip()

    first_line = next((line.strip() for line in draft_text.splitlines() if line.strip()), "")
    if not first_line:
        return "Untitled Draft"

    sanitized = WHITESPACE_RE.sub(" ", first_line).strip(" -:\t")
    if len(sanitized) <= 80:
        return sanitized
    return f"{sanitized[:77].rstrip()}..."
