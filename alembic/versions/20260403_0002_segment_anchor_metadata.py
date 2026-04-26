"""add segment anchor metadata

Revision ID: 20260403_0002
Revises: 20260331_0001
Create Date: 2026-04-03 00:02:00
"""

from __future__ import annotations

import re

from alembic import op
import sqlalchemy as sa


revision = "20260403_0002"
down_revision = "20260331_0001"
branch_labels = None
depends_on = None

DECLARATION_PARAGRAPH_RE = re.compile(r"¶\s*(\d+)")


def upgrade() -> None:
    op.add_column("segments", sa.Column("anchor_metadata", sa.JSON(), nullable=True))

    bind = op.get_bind()
    segments = sa.table(
        "segments",
        sa.column("id", sa.Uuid()),
        sa.column("source_document_id", sa.Uuid()),
        sa.column("page_start", sa.Integer()),
        sa.column("line_start", sa.Integer()),
        sa.column("page_end", sa.Integer()),
        sa.column("line_end", sa.Integer()),
        sa.column("speaker", sa.String()),
        sa.column("segment_type", sa.String()),
        sa.column("anchor_metadata", sa.JSON()),
    )
    source_documents = sa.table(
        "source_documents",
        sa.column("id", sa.Uuid()),
        sa.column("source_type", sa.String()),
    )

    rows = bind.execute(
        sa.select(
            segments.c.id,
            segments.c.page_start,
            segments.c.line_start,
            segments.c.page_end,
            segments.c.line_end,
            segments.c.speaker,
            segments.c.segment_type,
            source_documents.c.source_type,
        ).select_from(
            segments.join(source_documents, segments.c.source_document_id == source_documents.c.id)
        )
    ).mappings()

    for row in rows:
        anchor_metadata = _anchor_metadata_for_row(row)
        if anchor_metadata is None:
            continue
        bind.execute(
            sa.update(segments)
            .where(segments.c.id == row["id"])
            .values(anchor_metadata=anchor_metadata)
        )


def downgrade() -> None:
    op.drop_column("segments", "anchor_metadata")


def _anchor_metadata_for_row(row: sa.RowMapping) -> dict[str, object] | None:
    source_type = row["source_type"]
    if source_type == "DECLARATION":
        sequence_order = row["line_start"]
        if row["segment_type"] == "DECLARATION_PARAGRAPH":
            paragraph_number = _extract_paragraph_number(row["speaker"]) or sequence_order
            if paragraph_number is None:
                return None
            return {
                "kind": "paragraph",
                "paragraph_number": int(paragraph_number),
                "sequence_order": int(sequence_order or paragraph_number),
            }

        if row["segment_type"] == "DECLARATION_BLOCK":
            if sequence_order is None:
                return None
            return {
                "kind": "block",
                "block_index": int(sequence_order),
                "sequence_order": int(sequence_order),
            }

    if None not in {row["page_start"], row["line_start"], row["page_end"], row["line_end"]}:
        return {
            "kind": "page_line",
            "page_start": int(row["page_start"]),
            "line_start": int(row["line_start"]),
            "page_end": int(row["page_end"]),
            "line_end": int(row["line_end"]),
        }

    return None


def _extract_paragraph_number(speaker: str | None) -> int | None:
    if not speaker:
        return None
    match = DECLARATION_PARAGRAPH_RE.search(speaker)
    if match is None:
        return None
    return int(match.group(1))
