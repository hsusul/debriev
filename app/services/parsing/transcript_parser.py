"""Source-type-specific parsing strategies."""

from dataclasses import dataclass
import re

from app.core.enums import SourceType
from app.core.provenance import (
    build_block_anchor_metadata,
    build_exhibit_page_anchor_metadata,
    build_page_line_anchor_metadata,
    build_paragraph_anchor_metadata,
)
from app.services.parsing.base import ParsedSegment
from app.services.parsing.normalization import normalize_for_match

PAGE_HEADER_RE = re.compile(r"^\s*(?:page|pg\.?)\s+(\d+)\s*$", re.IGNORECASE)
LINE_RE = re.compile(r"^\s*(\d{1,3})\s+(.*\S.*)$")
INLINE_PAGE_LINE_RE = re.compile(r"^\s*(\d+):(\d+)\s+(.*\S.*)$")
SPEAKER_RE = re.compile(r"^(Q|A)\.\s", re.IGNORECASE)
LABEL_SPEAKER_RE = re.compile(r"^([A-Z][A-Z ]{1,40}):\s")
DECLARATION_NUMBER_RE = re.compile(
    r"^\s*(?:¶\s*(\d{1,4})|(?:\((\d{1,4})\)|(\d{1,4})[\.\)]))\s+(.*\S.*)$",
    re.IGNORECASE,
)
EXHIBIT_PAGE_HEADER_RE = re.compile(
    r"^\s*(?:exhibit(?:\s+[a-z0-9._-]+)?\s+)?page\s+(\d+)\s*$",
    re.IGNORECASE,
)
EXHIBIT_LABEL_RE = re.compile(r"^\s*([A-Z][A-Za-z0-9 /&()#._-]{0,40}):\s*(.*)$")

class TranscriptParser:
    """Very small deposition parser that preserves page/line anchors when present."""

    source_type = SourceType.DEPOSITION

    def parse(self, text: str) -> list[ParsedSegment]:
        current_page: int | None = None
        current_block: ParsedSegment | None = None
        fallback_lines: list[str] = []
        segments: list[ParsedSegment] = []

        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue

            page_match = PAGE_HEADER_RE.match(stripped)
            if page_match:
                current_page = int(page_match.group(1))
                continue

            line_number: int | None = None
            content = stripped

            inline_match = INLINE_PAGE_LINE_RE.match(stripped)
            if inline_match:
                current_page = int(inline_match.group(1))
                line_number = int(inline_match.group(2))
                content = inline_match.group(3).strip()
            else:
                line_match = LINE_RE.match(raw_line)
                if line_match and current_page is not None:
                    line_number = int(line_match.group(1))
                    content = line_match.group(2).strip()

            if line_number is None:
                if current_block:
                    segments.append(current_block)
                    current_block = None
                fallback_lines.append(stripped)
                continue

            if fallback_lines:
                segments.append(self._build_fallback_segment(fallback_lines))
                fallback_lines = []

            speaker = self._detect_speaker(content)
            if current_block and self._can_extend(current_block, current_page, line_number, speaker):
                current_block.page_end = current_page
                current_block.line_end = line_number
                if (
                    current_block.page_start is not None
                    and current_block.line_start is not None
                    and current_page is not None
                ):
                    current_block.anchor_metadata = build_page_line_anchor_metadata(
                        page_start=current_block.page_start,
                        line_start=current_block.line_start,
                        page_end=current_page,
                        line_end=line_number,
                    )
                current_block.raw_text = f"{current_block.raw_text}\n{content}"
                current_block.normalized_text = normalize_for_match(current_block.raw_text)
                continue

            if current_block:
                segments.append(current_block)

            current_block = ParsedSegment(
                page_start=current_page,
                line_start=line_number,
                page_end=current_page,
                line_end=line_number,
                anchor_metadata=build_page_line_anchor_metadata(
                    page_start=current_page,
                    line_start=line_number,
                    page_end=current_page,
                    line_end=line_number,
                ) if current_page is not None else None,
                raw_text=content,
                normalized_text=normalize_for_match(content),
                speaker=speaker,
                segment_type=self._segment_type_for_speaker(speaker),
            )

        if current_block:
            segments.append(current_block)

        if fallback_lines:
            segments.append(self._build_fallback_segment(fallback_lines))

        return segments

    def confidence_for_segments(self, segments: list[ParsedSegment]) -> float:
        return 0.95 if any(segment.has_usable_anchor for segment in segments) else 0.55

    def _can_extend(
        self,
        current_block: ParsedSegment,
        page_number: int | None,
        line_number: int,
        speaker: str | None,
    ) -> bool:
        return (
            current_block.page_end == page_number
            and current_block.line_end is not None
            and current_block.line_end + 1 == line_number
            and current_block.speaker == speaker
        )

    def _detect_speaker(self, text: str) -> str | None:
        if speaker_match := SPEAKER_RE.match(text):
            return speaker_match.group(1).upper()
        if label_match := LABEL_SPEAKER_RE.match(text):
            return label_match.group(1).strip()
        return None

    def _segment_type_for_speaker(self, speaker: str | None) -> str:
        if speaker == "Q":
            return "QUESTION_BLOCK"
        if speaker == "A":
            return "ANSWER_BLOCK"
        return "TRANSCRIPT_BLOCK"

    def _build_fallback_segment(self, lines: list[str]) -> ParsedSegment:
        fallback_text = "\n".join(lines)
        return ParsedSegment(
            page_start=None,
            line_start=None,
            page_end=None,
            line_end=None,
            anchor_metadata=None,
            raw_text=fallback_text,
            normalized_text=normalize_for_match(fallback_text),
            speaker=None,
            segment_type="UNANCHORED_TEXT",
        )


@dataclass(slots=True)
class _DeclarationBlock:
    content: str
    paragraph_number: int | None
    sequence_order: int


class DeclarationParser:
    """Small declaration parser that emits stable paragraph-like segments."""

    source_type = SourceType.DECLARATION

    def parse(self, text: str) -> list[ParsedSegment]:
        blocks = self._collect_blocks(text)
        return [
            ParsedSegment(
                page_start=None,
                line_start=None,
                page_end=None,
                line_end=None,
                anchor_metadata=self._anchor_metadata_for_block(block),
                raw_text=block.content,
                normalized_text=normalize_for_match(block.content),
                speaker=self._speaker_for_block(block),
                segment_type=self._segment_type_for_block(block),
            )
            for block in blocks
        ]

    def _collect_blocks(self, text: str) -> list[_DeclarationBlock]:
        blocks: list[_DeclarationBlock] = []
        current_lines: list[str] = []
        current_paragraph_number: int | None = None
        sequence_order = 1

        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                if current_lines:
                    blocks.append(
                        _DeclarationBlock(
                            content="\n".join(current_lines),
                            paragraph_number=current_paragraph_number,
                            sequence_order=sequence_order,
                        )
                    )
                    sequence_order += 1
                    current_lines = []
                    current_paragraph_number = None
                continue

            paragraph_match = DECLARATION_NUMBER_RE.match(stripped)
            if paragraph_match is not None:
                if current_lines:
                    blocks.append(
                        _DeclarationBlock(
                            content="\n".join(current_lines),
                            paragraph_number=current_paragraph_number,
                            sequence_order=sequence_order,
                        )
                    )
                    sequence_order += 1

                paragraph_number = next(
                    int(group)
                    for group in paragraph_match.groups()[:3]
                    if group is not None
                )
                current_lines = [paragraph_match.group(4).strip()]
                current_paragraph_number = paragraph_number
                continue

            if current_lines:
                current_lines.append(stripped)
            else:
                current_lines = [stripped]
                current_paragraph_number = None

        if current_lines:
            blocks.append(
                _DeclarationBlock(
                    content="\n".join(current_lines),
                    paragraph_number=current_paragraph_number,
                    sequence_order=sequence_order,
                )
            )

        return blocks

    def _speaker_for_block(self, block: _DeclarationBlock) -> str:
        if block.paragraph_number is None:
            return "DECLARANT"
        return f"DECLARANT ¶{block.paragraph_number}"

    def _segment_type_for_block(self, block: _DeclarationBlock) -> str:
        if block.paragraph_number is None:
            return "DECLARATION_BLOCK"
        return "DECLARATION_PARAGRAPH"

    def _anchor_metadata_for_block(self, block: _DeclarationBlock) -> dict[str, object]:
        if block.paragraph_number is None:
            return build_block_anchor_metadata(
                block_index=block.sequence_order,
                sequence_order=block.sequence_order,
            )
        return build_paragraph_anchor_metadata(
            paragraph_number=block.paragraph_number,
            sequence_order=block.sequence_order,
        )

    def confidence_for_segments(self, segments: list[ParsedSegment]) -> float:
        has_numbered_paragraphs = any(segment.segment_type == "DECLARATION_PARAGRAPH" for segment in segments)
        return 0.85 if has_numbered_paragraphs else 0.7


@dataclass(slots=True)
class _ExhibitBlock:
    content: str
    label: str | None
    page_number: int | None
    page_block_index: int | None
    sequence_order: int


class ExhibitParser:
    """Text-first exhibit parser for page blocks, labeled blocks, and fallback text blocks."""

    source_type = SourceType.EXHIBIT

    def parse(self, text: str) -> list[ParsedSegment]:
        blocks: list[_ExhibitBlock] = []
        current_lines: list[str] = []
        current_label: str | None = None
        current_label_allows_continuation = False
        current_page: int | None = None
        page_block_index = 0
        sequence_order = 1

        def flush_current_block() -> None:
            nonlocal current_lines, current_label, current_label_allows_continuation, page_block_index, sequence_order
            if not current_lines:
                return

            content = "\n".join(current_lines).strip()
            if not content:
                current_lines = []
                current_label = None
                return

            block_index: int | None = None
            if current_page is not None:
                page_block_index += 1
                block_index = page_block_index

            blocks.append(
                _ExhibitBlock(
                    content=content,
                    label=current_label,
                    page_number=current_page,
                    page_block_index=block_index,
                    sequence_order=sequence_order,
                )
            )
            sequence_order += 1
            current_lines = []
            current_label = None
            current_label_allows_continuation = False

        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                flush_current_block()
                continue

            page_match = EXHIBIT_PAGE_HEADER_RE.match(stripped)
            if page_match is not None:
                flush_current_block()
                current_page = int(page_match.group(1))
                page_block_index = 0
                continue

            label_match = EXHIBIT_LABEL_RE.match(stripped)
            if label_match is not None:
                flush_current_block()
                current_label = label_match.group(1).strip()
                label_value = label_match.group(2).strip()
                current_label_allows_continuation = not bool(label_value)
                current_lines = [f"{current_label}: {label_value}".rstrip(": ").strip()]
                continue

            if current_label is not None and not current_label_allows_continuation:
                flush_current_block()

            if current_lines:
                current_lines.append(stripped)
            else:
                current_lines = [stripped]

        flush_current_block()

        return [
            ParsedSegment(
                page_start=None,
                line_start=None,
                page_end=None,
                line_end=None,
                anchor_metadata=self._anchor_metadata_for_block(block),
                raw_text=block.content,
                normalized_text=normalize_for_match(block.content),
                speaker=block.label,
                segment_type=self._segment_type_for_block(block),
            )
            for block in blocks
        ]

    def confidence_for_segments(self, segments: list[ParsedSegment]) -> float:
        if any(segment.anchor_metadata and segment.anchor_metadata.get("kind") == "exhibit_page" for segment in segments):
            return 0.8
        if any(segment.segment_type == "EXHIBIT_LABELED_BLOCK" for segment in segments):
            return 0.72
        return 0.65

    def _segment_type_for_block(self, block: _ExhibitBlock) -> str:
        if block.label is not None:
            return "EXHIBIT_LABELED_BLOCK"
        if block.page_number is not None:
            return "EXHIBIT_PAGE_BLOCK"
        return "EXHIBIT_TEXT_BLOCK"

    def _anchor_metadata_for_block(self, block: _ExhibitBlock) -> dict[str, object]:
        if block.page_number is not None:
            return build_exhibit_page_anchor_metadata(
                page_number=block.page_number,
                block_index=block.page_block_index,
                sequence_order=block.sequence_order,
                label=block.label,
            )
        return build_block_anchor_metadata(
            block_index=block.sequence_order,
            sequence_order=block.sequence_order,
            label=block.label,
        )
