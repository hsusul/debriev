# Debriev

Debriev is a monorepo scaffold for legal citation integrity reporting.

## v1 Scope
- US case citations only (`U.S.`, `F.2d`/`F.3d`, `F. Supp.`, `S. Ct.`, `L. Ed.`).
- No statutes, regulations, or Bluebook edge cases in v1.

## Implemented
- Task 2: PDF text extraction + regex-based citation extraction in `packages/core`.
- Day 3: CourtListener-backed verification for `U.S.` citations.
- Day 3 UI: report page renders source PDF and highlights extracted citations.

## Run with Docker Compose
```bash
docker compose -f infra/compose.yml up --build
```

Services:
- Postgres: `localhost:5432`
- API: `localhost:8000`
- Web: `localhost:3000`

## Local Development

### Two-terminal workflow

Terminal 1:
```bash
cd apps/api
./.venv/bin/python -m uvicorn app.main:app --reload --port 8000
```

Terminal 2:
```bash
cd apps/web
npm run dev
```

Optional worker terminal:
```bash
cd apps/api
./.venv/bin/python -m app.worker
```

### One-command workflow (repo root)

```bash
npm run dev:all
```

Additional root scripts:
- `npm run dev:web`
- `npm run dev:api`
- `npm run dev:worker`

## Smoke Test
1. Open `http://localhost:3000/upload`.
2. Upload a PDF containing case citations.
3. Open `/reports/{doc_id}` and confirm:
   - PDF renders in the center panel.
   - Citation list appears on the right panel.
   - Clicking a citation jumps to the matched page and highlights text.
   - Verify button updates statuses and links.

## API Endpoints
- `GET /health`
- `POST /v1/upload` (multipart field `file`, only `.pdf`)
- `GET /v1/documents`
- `GET /v1/documents/{doc_id}/pdf`
- `GET /v1/citations/{doc_id}`
- `POST /v1/verify/{doc_id}`
- `GET /v1/reports/{doc_id}`
- `POST /v1/retrieval/query`

## Local Core Tests
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e packages/core[test]
pytest packages/core/tests
```

## Environment Variables
API:
- `DATABASE_URL` (default `postgresql+psycopg://debriev:debrievdev@postgres:5432/debriev`)
- `STORAGE_DIR` (default `/tmp/debriev_uploads`)
- `RETRIEVAL_DB_PATH` (default `./data/retrieval.db`)
- `EMBED_PROVIDER` (`stub` or `openai`, default `stub`)
- `OPENAI_API_KEY` (required when `EMBED_PROVIDER=openai`)

Web:
- `NEXT_PUBLIC_API_BASE_URL` (default `http://localhost:8000`)
- `API_BASE_URL` (default `http://api:8000`)

## Indexing & Retrieval (MVP)
The retrieval MVP stores chunk text and embeddings in SQLite, then does cosine similarity search.

1. Index a document:
```bash
python scripts/index_doc.py --doc-id <doc_uuid>
```

2. Query indexed chunks:
```bash
curl -s -X POST http://localhost:8000/v1/retrieval/query \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"<doc_uuid>","query":"What does this document say about jurisdiction?","k":3}'
```

Notes:
- If `OPENAI_API_KEY` is not set, embeddings default to deterministic stub vectors.
- If `EMBED_PROVIDER=openai` and key is set, OpenAI embeddings are used.
