"""Source-agnostic anchor and provenance helpers."""

from collections.abc import Mapping
from datetime import datetime
from typing import Any
from uuid import UUID

PAGE_LINE_KIND = "page_line"
PARAGRAPH_KIND = "paragraph"
BLOCK_KIND = "block"
EXHIBIT_PAGE_KIND = "exhibit_page"


def build_page_line_anchor_metadata(
    *,
    page_start: int,
    line_start: int,
    page_end: int,
    line_end: int,
) -> dict[str, object]:
    return {
        "kind": PAGE_LINE_KIND,
        "page_start": page_start,
        "line_start": line_start,
        "page_end": page_end,
        "line_end": line_end,
    }


def build_paragraph_anchor_metadata(
    *,
    paragraph_number: int,
    sequence_order: int | None = None,
) -> dict[str, object]:
    return {
        "kind": PARAGRAPH_KIND,
        "paragraph_number": paragraph_number,
        "sequence_order": sequence_order if sequence_order is not None else paragraph_number,
    }


def build_block_anchor_metadata(
    *,
    block_index: int,
    sequence_order: int | None = None,
    label: str | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "kind": BLOCK_KIND,
        "block_index": block_index,
        "sequence_order": sequence_order if sequence_order is not None else block_index,
    }
    normalized_label = _coerce_label(label)
    if normalized_label is not None:
        metadata["label"] = normalized_label
    return metadata


def build_exhibit_page_anchor_metadata(
    *,
    page_number: int,
    block_index: int | None = None,
    sequence_order: int | None = None,
    label: str | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "kind": EXHIBIT_PAGE_KIND,
        "page_number": page_number,
    }
    if block_index is not None:
        metadata["block_index"] = block_index
    if sequence_order is not None:
        metadata["sequence_order"] = sequence_order
    elif block_index is not None:
        metadata["sequence_order"] = block_index
    else:
        metadata["sequence_order"] = page_number
    normalized_label = _coerce_label(label)
    if normalized_label is not None:
        metadata["label"] = normalized_label
    return metadata


def normalize_anchor_metadata(
    anchor_metadata: Mapping[str, Any] | None,
    *,
    page_start: int | None = None,
    line_start: int | None = None,
    page_end: int | None = None,
    line_end: int | None = None,
) -> dict[str, object] | None:
    if anchor_metadata:
        kind = str(anchor_metadata.get("kind") or "").lower()
        if kind == PAGE_LINE_KIND:
            normalized = _normalize_page_line_anchor(anchor_metadata)
            if normalized is not None:
                return normalized
        elif kind == PARAGRAPH_KIND:
            normalized = _normalize_paragraph_anchor(anchor_metadata)
            if normalized is not None:
                return normalized
        elif kind == BLOCK_KIND:
            normalized = _normalize_block_anchor(anchor_metadata)
            if normalized is not None:
                return normalized
        elif kind == EXHIBIT_PAGE_KIND:
            normalized = _normalize_exhibit_page_anchor(anchor_metadata)
            if normalized is not None:
                return normalized

    if None not in {page_start, line_start, page_end, line_end}:
        return build_page_line_anchor_metadata(
            page_start=int(page_start),
            line_start=int(line_start),
            page_end=int(page_end),
            line_end=int(line_end),
        )

    return None


def render_anchor_text(
    anchor_metadata: Mapping[str, Any] | None,
    *,
    page_start: int | None = None,
    line_start: int | None = None,
    page_end: int | None = None,
    line_end: int | None = None,
) -> str:
    normalized = normalize_anchor_metadata(
        anchor_metadata,
        page_start=page_start,
        line_start=line_start,
        page_end=page_end,
        line_end=line_end,
    )
    if normalized is None:
        return "unanchored segment"

    kind = str(normalized["kind"])
    if kind == PAGE_LINE_KIND:
        return (
            f"p.{normalized['page_start']}:{normalized['line_start']}"
            f"-{normalized['page_end']}:{normalized['line_end']}"
        )
    if kind == PARAGRAPH_KIND:
        return f"¶{normalized['paragraph_number']}"
    if kind == BLOCK_KIND:
        base = f"block {normalized['block_index']}"
        label = normalized.get("label")
        if label:
            return f"{base} [{label}]"
        return base
    if kind == EXHIBIT_PAGE_KIND:
        base = f"ex. p.{normalized['page_number']}"
        block_index = normalized.get("block_index")
        if block_index is not None:
            base = f"{base} block {block_index}"
        label = normalized.get("label")
        if label:
            base = f"{base} [{label}]"
        return base
    return "unanchored segment"


def has_usable_anchor(
    anchor_metadata: Mapping[str, Any] | None,
    *,
    page_start: int | None = None,
    line_start: int | None = None,
    page_end: int | None = None,
    line_end: int | None = None,
) -> bool:
    normalized = normalize_anchor_metadata(
        anchor_metadata,
        page_start=page_start,
        line_start=line_start,
        page_end=page_end,
        line_end=line_end,
    )
    if normalized is None:
        return False

    kind = str(normalized["kind"])
    if kind == PAGE_LINE_KIND:
        start = (int(normalized["page_start"]), int(normalized["line_start"]))
        end = (int(normalized["page_end"]), int(normalized["line_end"]))
        return end >= start
    if kind == PARAGRAPH_KIND:
        return int(normalized["paragraph_number"]) > 0
    if kind == BLOCK_KIND:
        return int(normalized["block_index"]) > 0
    if kind == EXHIBIT_PAGE_KIND:
        page_number = int(normalized["page_number"])
        block_index = _coerce_int(normalized.get("block_index"))
        if block_index is None:
            return page_number > 0
        return page_number > 0 and block_index > 0
    return False


def anchor_sort_key(
    anchor_metadata: Mapping[str, Any] | None,
    *,
    page_start: int | None = None,
    line_start: int | None = None,
    page_end: int | None = None,
    line_end: int | None = None,
    created_at: datetime | None = None,
    segment_id: UUID | None = None,
) -> tuple[object, ...]:
    normalized = normalize_anchor_metadata(
        anchor_metadata,
        page_start=page_start,
        line_start=line_start,
        page_end=page_end,
        line_end=line_end,
    )
    created_marker = created_at or datetime.min
    id_marker = str(segment_id or "")

    if normalized is None:
        return (99, float("inf"), float("inf"), float("inf"), float("inf"), created_marker, id_marker)

    kind = str(normalized["kind"])
    if kind == PAGE_LINE_KIND:
        return (
            0,
            int(normalized["page_start"]),
            int(normalized["line_start"]),
            int(normalized["page_end"]),
            int(normalized["line_end"]),
            created_marker,
            id_marker,
        )

    if kind == EXHIBIT_PAGE_KIND:
        sequence_order = _coerce_int(normalized.get("sequence_order"))
        if sequence_order is not None:
            return (1, sequence_order, 0, 0, 0, created_marker, id_marker)
        return (
            1,
            int(normalized["page_number"]),
            _coerce_int(normalized.get("block_index")) or 0,
            0,
            0,
            created_marker,
            id_marker,
        )

    sequence_order = _coerce_int(normalized.get("sequence_order"))
    if sequence_order is not None:
        return (2, sequence_order, 0, 0, 0, created_marker, id_marker)

    if kind == PARAGRAPH_KIND:
        return (2, int(normalized["paragraph_number"]), 0, 0, 0, created_marker, id_marker)
    if kind == BLOCK_KIND:
        return (2, int(normalized["block_index"]), 0, 0, 0, created_marker, id_marker)
    return (98, float("inf"), float("inf"), float("inf"), float("inf"), created_marker, id_marker)


def _normalize_page_line_anchor(anchor_metadata: Mapping[str, Any]) -> dict[str, object] | None:
    page_start = _coerce_int(anchor_metadata.get("page_start"))
    line_start = _coerce_int(anchor_metadata.get("line_start"))
    page_end = _coerce_int(anchor_metadata.get("page_end"))
    line_end = _coerce_int(anchor_metadata.get("line_end"))
    if None in {page_start, line_start, page_end, line_end}:
        return None
    return build_page_line_anchor_metadata(
        page_start=page_start,
        line_start=line_start,
        page_end=page_end,
        line_end=line_end,
    )


def _normalize_paragraph_anchor(anchor_metadata: Mapping[str, Any]) -> dict[str, object] | None:
    paragraph_number = _coerce_int(anchor_metadata.get("paragraph_number"))
    if paragraph_number is None:
        return None
    return build_paragraph_anchor_metadata(
        paragraph_number=paragraph_number,
        sequence_order=_coerce_int(anchor_metadata.get("sequence_order")),
    )


def _normalize_block_anchor(anchor_metadata: Mapping[str, Any]) -> dict[str, object] | None:
    block_index = _coerce_int(anchor_metadata.get("block_index"))
    if block_index is None:
        return None
    return build_block_anchor_metadata(
        block_index=block_index,
        sequence_order=_coerce_int(anchor_metadata.get("sequence_order")),
        label=_coerce_label(anchor_metadata.get("label")),
    )


def _normalize_exhibit_page_anchor(anchor_metadata: Mapping[str, Any]) -> dict[str, object] | None:
    page_number = _coerce_int(anchor_metadata.get("page_number"))
    if page_number is None:
        return None
    return build_exhibit_page_anchor_metadata(
        page_number=page_number,
        block_index=_coerce_int(anchor_metadata.get("block_index")),
        sequence_order=_coerce_int(anchor_metadata.get("sequence_order")),
        label=_coerce_label(anchor_metadata.get("label")),
    )


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _coerce_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None
