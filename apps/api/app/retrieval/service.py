from __future__ import annotations

from typing import Any
from uuid import UUID

try:
    from sqlmodel import Session, select
except ModuleNotFoundError:  # pragma: no cover - allows extractor self-tests without db deps
    Session = Any  # type: ignore[assignment]

    def select(*args: Any, **kwargs: Any) -> Any:
        raise ModuleNotFoundError("sqlmodel is required for database-backed retrieval indexing")

from app.db import Document

from .chunking import chunk_document_text
from .embeddings import cosine_similarity, embed_texts
from .store import (
    load_document_chunks_with_embeddings,
    make_chunk_id,
    replace_document_chunks,
    upsert_chunk_embeddings,
)


def index_document_from_db(
    *,
    session: Session,
    doc_id: UUID,
    db_path: str,
    provider: str,
    openai_api_key: str | None,
) -> int:
    document = session.exec(select(Document).where(Document.doc_id == doc_id)).first()
    if document is None:
        raise ValueError(f"Document {doc_id} was not found")

    text = (document.stub_text or "").strip()
    if not text:
        raise ValueError(f"Document {doc_id} has no extracted text to index")

    chunks = chunk_document_text(doc_id=str(doc_id), text=text)
    replace_document_chunks(db_path=db_path, doc_id=str(doc_id), chunks=chunks)

    embeddings = embed_texts(
        [chunk.text for chunk in chunks],
        provider=provider,
        openai_api_key=openai_api_key,
    )
    by_chunk_id = {
        make_chunk_id(doc_id=str(doc_id), chunk_index=chunk.chunk_index): embedding
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    }
    upsert_chunk_embeddings(db_path=db_path, doc_id=str(doc_id), embeddings_by_chunk_id=by_chunk_id)

    return len(chunks)


def query_document_chunks(
    *,
    db_path: str,
    doc_id: str,
    query: str,
    k: int,
    provider: str,
    openai_api_key: str | None,
) -> list[dict[str, object]]:
    raw_rows = load_document_chunks_with_embeddings(db_path=db_path, doc_id=doc_id)
    if not raw_rows:
        return []

    normalized_rows = [
        {
            "chunk_id": str(row["chunk_id"]),
            "chunk_index": int(row["chunk_index"]),
            "page": int(row["page"]) if row["page"] is not None else None,
            "text": str(row["text"]),
            "embedding": row["embedding"],
        }
        for row in raw_rows
    ]

    query_embedding = embed_texts([query], provider=provider, openai_api_key=openai_api_key)[0]
    scored = [
        {
            "chunk_id": row["chunk_id"],
            "chunk_index": row["chunk_index"],
            "score": cosine_similarity(query_embedding, row["embedding"]),  # type: ignore[arg-type]
            "page": row["page"],
            "text": row["text"],
        }
        for row in normalized_rows
    ]
    scored.sort(key=lambda item: float(item["score"]), reverse=True)

    top_hits = scored[: max(1, k)]
    by_index = {row["chunk_index"]: row for row in normalized_rows}

    expanded: list[dict[str, object]] = []
    seen_chunk_ids: set[str] = set()

    # Group by hit rank: hit first, then neighbors (-1, +1), deduped globally.
    for hit in top_hits:
        hit_chunk_id = str(hit["chunk_id"])
        hit_score = float(hit["score"])
        hit_index = int(hit["chunk_index"])

        if hit_chunk_id not in seen_chunk_ids:
            expanded.append(
                {
                    "chunk_id": hit_chunk_id,
                    "score": hit_score,
                    "page": hit["page"],
                    "text": hit["text"],
                }
            )
            seen_chunk_ids.add(hit_chunk_id)

        for neighbor_index in (hit_index - 1, hit_index + 1):
            neighbor = by_index.get(neighbor_index)
            if neighbor is None:
                continue

            neighbor_chunk_id = str(neighbor["chunk_id"])
            if neighbor_chunk_id in seen_chunk_ids:
                continue

            expanded.append(
                {
                    "chunk_id": neighbor_chunk_id,
                    "score": hit_score,
                    "page": neighbor["page"],
                    "text": neighbor["text"],
                }
            )
            seen_chunk_ids.add(neighbor_chunk_id)

    return expanded


def query_project_chunks(
    *,
    db_path: str,
    doc_ids: list[str],
    query: str,
    k_per_doc: int,
    k_total: int,
    provider: str,
    openai_api_key: str | None,
) -> list[dict[str, object]]:
    if not doc_ids:
        return []

    per_doc_k = max(1, min(k_per_doc, 20))
    total_k = max(1, min(k_total, 20))

    merged: list[dict[str, object]] = []
    for doc_id in doc_ids:
        rows = query_document_chunks(
            db_path=db_path,
            doc_id=doc_id,
            query=query,
            k=per_doc_k,
            provider=provider,
            openai_api_key=openai_api_key,
        )
        for row in rows:
            merged.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": row["chunk_id"],
                    "score": float(row["score"]),
                    "page": row["page"],
                    "text": row["text"],
                }
            )

    merged.sort(key=lambda item: float(item["score"]), reverse=True)
    return merged[:total_k]


def _test_neighbor_expansion_order() -> None:
    """Tiny manual sanity check for grouped ordering and de-dup behavior."""
    ranked = [10, 11]
    seen: set[int] = set()
    ordered: list[int] = []
    for idx in ranked:
        for candidate in (idx, idx - 1, idx + 1):
            if candidate in seen:
                continue
            seen.add(candidate)
            ordered.append(candidate)
    assert ordered == [10, 9, 11, 12]
