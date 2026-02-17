# Debriev

Debriev is a monorepo scaffold for legal citation integrity reporting.

## v1 Scope
- US case citations only (`U.S.`, `F.2d`/`F.3d`, `F. Supp.`, `S. Ct.`, `L. Ed.`).
- No statutes, regulations, or Bluebook edge cases in v1.
- Verification remains stubbed.

## Implemented in Task 2
- PDF text extraction via PyMuPDF.
- Regex-based US case citation extraction in `packages/core`.
- API upload flow now stores extracted citation spans and returns a report containing citations.

## Run with Docker Compose
```bash
docker compose -f infra/compose.yml up --build
```

Services:
- Postgres: `localhost:5432`
- API: `localhost:8000`
- Web: `localhost:3000`

## Demo Task 2
1. Open `http://localhost:3000/upload`.
2. Upload a PDF containing US case citations (for example: `410 U.S. 113`, `123 F.3d 456`).
3. After redirect, inspect `/reports/{doc_id}` JSON and confirm extracted citation entries.
4. Optionally open `http://localhost:3000/documents` to list uploaded documents.

## API Endpoints
- `GET /health`
- `POST /v1/upload` (multipart field `file`, only `.pdf`)
- `GET /v1/documents`
- `GET /v1/reports/{doc_id}`

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

Web:
- `NEXT_PUBLIC_API_BASE_URL` (default `http://localhost:8000`)
- `API_BASE_URL` (default `http://api:8000`)
