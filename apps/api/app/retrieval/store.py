from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from .chunking import TextChunk


def init_retrieval_db(db_path: str) -> None:
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                doc_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                page INTEGER,
                text TEXT NOT NULL,
                PRIMARY KEY (doc_id, chunk_id)
            );

            CREATE INDEX IF NOT EXISTS idx_document_chunks_doc_index
                ON document_chunks (doc_id, chunk_index);

            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                doc_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                PRIMARY KEY (doc_id, chunk_id),
                FOREIGN KEY (doc_id, chunk_id)
                  REFERENCES document_chunks(doc_id, chunk_id)
                  ON DELETE CASCADE
            );
            """
        )
        conn.commit()


def replace_document_chunks(db_path: str, doc_id: str, chunks: Iterable[TextChunk]) -> None:
    rows = [
        (
            doc_id,
            make_chunk_id(doc_id=doc_id, chunk_index=chunk.chunk_index),
            chunk.chunk_index,
            chunk.page,
            chunk.text,
        )
        for chunk in chunks
    ]

    with _connect(db_path) as conn:
        conn.execute("DELETE FROM chunk_embeddings WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM document_chunks WHERE doc_id = ?", (doc_id,))
        if rows:
            conn.executemany(
                """
                INSERT INTO document_chunks (doc_id, chunk_id, chunk_index, page, text)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
        conn.commit()


def upsert_chunk_embeddings(db_path: str, doc_id: str, embeddings_by_chunk_id: dict[str, list[float]]) -> None:
    rows = [
        (doc_id, chunk_id, json.dumps(embedding))
        for chunk_id, embedding in embeddings_by_chunk_id.items()
    ]
    if not rows:
        return

    with _connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO chunk_embeddings (doc_id, chunk_id, embedding_json)
            VALUES (?, ?, ?)
            ON CONFLICT(doc_id, chunk_id) DO UPDATE SET embedding_json = excluded.embedding_json
            """,
            rows,
        )
        conn.commit()


def load_document_chunks_with_embeddings(db_path: str, doc_id: str) -> list[dict[str, object]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT dc.chunk_id, dc.chunk_index, dc.page, dc.text, ce.embedding_json
            FROM document_chunks dc
            JOIN chunk_embeddings ce
              ON ce.doc_id = dc.doc_id
             AND ce.chunk_id = dc.chunk_id
            WHERE dc.doc_id = ?
            ORDER BY dc.chunk_index ASC
            """,
            (doc_id,),
        ).fetchall()

    parsed_rows: list[dict[str, object]] = []
    for row in rows:
        try:
            embedding_raw = json.loads(row["embedding_json"])
        except json.JSONDecodeError:
            continue
        if not isinstance(embedding_raw, list):
            continue

        embedding: list[float] = []
        for value in embedding_raw:
            if isinstance(value, (int, float)):
                embedding.append(float(value))

        parsed_rows.append(
            {
                "chunk_id": row["chunk_id"],
                "chunk_index": int(row["chunk_index"]),
                "page": row["page"],
                "text": row["text"],
                "embedding": embedding,
            }
        )

    return parsed_rows


def make_chunk_id(*, doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}:chunk:{chunk_index}"


def _connect(db_path: str) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn
