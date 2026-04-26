"""Parser primitives shared across source-specific parsing strategies."""

from dataclasses import asdict, dataclass
from typing import Protocol

from app.core.enums import SourceType
from app.core.provenance import has_usable_anchor, normalize_anchor_metadata, render_anchor_text


@dataclass(slots=True)
class ParsedSegment:
    """Parsed source segment with normalized provenance accessors."""

    page_start: int | None
    line_start: int | None
    page_end: int | None
    line_end: int | None
    anchor_metadata: dict[str, object] | None
    raw_text: str
    normalized_text: str
    speaker: str | None
    segment_type: str

    def to_record(self) -> dict[str, object]:
        return asdict(self)

    @property
    def normalized_anchor_metadata(self) -> dict[str, object] | None:
        return normalize_anchor_metadata(
            self.anchor_metadata,
            page_start=self.page_start,
            line_start=self.line_start,
            page_end=self.page_end,
            line_end=self.line_end,
        )

    @property
    def rendered_anchor(self) -> str:
        return render_anchor_text(
            self.anchor_metadata,
            page_start=self.page_start,
            line_start=self.line_start,
            page_end=self.page_end,
            line_end=self.line_end,
        )

    @property
    def has_usable_anchor(self) -> bool:
        return has_usable_anchor(
            self.anchor_metadata,
            page_start=self.page_start,
            line_start=self.line_start,
            page_end=self.page_end,
            line_end=self.line_end,
        )


class SourceParser(Protocol):
    """Minimal parser protocol for source-type-specific parsing strategies."""

    source_type: SourceType

    def parse(self, text: str) -> list[ParsedSegment]:
        """Parse raw source text into normalized segment payloads."""

    def confidence_for_segments(self, segments: list[ParsedSegment]) -> float:
        """Estimate parser confidence for the parsed segment set."""
