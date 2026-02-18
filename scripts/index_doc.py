#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from uuid import UUID


REPO_ROOT = Path(__file__).resolve().parents[1]
API_PACKAGE_ROOT = REPO_ROOT / "apps" / "api"
if str(API_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(API_PACKAGE_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index a single document for retrieval.")
    parser.add_argument("--doc-id", required=True, help="Document UUID to index")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        doc_id = UUID(args.doc_id)
    except ValueError:
        print("Invalid --doc-id. Expected UUID.", file=sys.stderr)
        return 2

    from sqlmodel import Session

    from app.db.session import engine
    from app.retrieval.service import index_document_from_db
    from app.retrieval.store import init_retrieval_db
    from app.settings import settings

    init_retrieval_db(settings.retrieval_db_path)

    try:
        with Session(engine) as session:
            chunk_count = index_document_from_db(
                session=session,
                doc_id=doc_id,
                db_path=settings.retrieval_db_path,
                provider=settings.embed_provider,
                openai_api_key=settings.openai_api_key,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"Indexing failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Indexed {chunk_count} chunk(s) for doc_id={doc_id} "
        f"into {settings.retrieval_db_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
