from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence


@dataclass(frozen=True)
class TextChunk:
    doc_id: str
    page: int | None
    chunk_index: int
    text: str


def chunk_document_text(
    doc_id: str,
    text: str,
    page_texts: Sequence[str] | None = None,
    *,
    target_size: int = 1000,
    min_size: int = 800,
    max_size: int = 1200,
    overlap: int = 150,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    chunk_index = 0

    if page_texts:
        for page_number, page_text in enumerate(page_texts, start=1):
            normalized = _normalize_text(page_text)
            if not normalized:
                continue
            page_chunks = _chunk_single_text(
                doc_id=doc_id,
                text=normalized,
                page=page_number,
                start_chunk_index=chunk_index,
                target_size=target_size,
                min_size=min_size,
                max_size=max_size,
                overlap=overlap,
            )
            chunks.extend(page_chunks)
            chunk_index += len(page_chunks)
        return chunks

    normalized = _normalize_text(text)
    if not normalized:
        return []

    return _chunk_single_text(
        doc_id=doc_id,
        text=normalized,
        page=None,
        start_chunk_index=0,
        target_size=target_size,
        min_size=min_size,
        max_size=max_size,
        overlap=overlap,
    )


def _chunk_single_text(
    *,
    doc_id: str,
    text: str,
    page: int | None,
    start_chunk_index: int,
    target_size: int,
    min_size: int,
    max_size: int,
    overlap: int,
) -> list[TextChunk]:
    results: list[TextChunk] = []
    text_length = len(text)
    cursor = 0
    chunk_index = start_chunk_index

    while cursor < text_length:
        if text_length - cursor <= max_size:
            end = text_length
        else:
            end = _pick_breakpoint(
                text=text,
                start=cursor,
                target_size=target_size,
                min_size=min_size,
                max_size=max_size,
            )

        value = text[cursor:end].strip()
        if value:
            results.append(
                TextChunk(
                    doc_id=doc_id,
                    page=page,
                    chunk_index=chunk_index,
                    text=value,
                )
            )
            chunk_index += 1

        if end >= text_length:
            break

        next_cursor = max(end - overlap, cursor + 1)
        if next_cursor <= cursor:
            break
        cursor = next_cursor

    return results


def _pick_breakpoint(
    *,
    text: str,
    start: int,
    target_size: int,
    min_size: int,
    max_size: int,
) -> int:
    text_length = len(text)
    min_end = min(text_length, start + min_size)
    ideal_end = min(text_length, start + target_size)
    max_end = min(text_length, start + max_size)

    if min_end >= text_length:
        return text_length

    paragraph_breaks = [
        match.end()
        for match in re.finditer(r"\n\s*\n", text[min_end:max_end])
    ]
    if paragraph_breaks:
        return min(
            (min_end + offset for offset in paragraph_breaks),
            key=lambda idx: abs(idx - ideal_end),
        )

    line_breaks = [match.end() for match in re.finditer(r"\n", text[min_end:max_end])]
    if line_breaks:
        return min(
            (min_end + offset for offset in line_breaks),
            key=lambda idx: abs(idx - ideal_end),
        )

    for idx in range(ideal_end, min_end, -1):
        if text[idx - 1].isspace():
            return idx

    for idx in range(ideal_end, max_end):
        if text[idx].isspace():
            return idx + 1

    return max_end


def _normalize_text(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()
