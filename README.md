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
