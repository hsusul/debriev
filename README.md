# Debriev

Debriev is a backend-first litigation drafting engine for validating factual claim units against anchored evidentiary record segments. This scaffold is intentionally narrow and deposition-first: transcript ingestion, anchored segment parsing, draft/assertion storage, lightweight claim extraction, hybrid verification, and immutable audit persistence.

## What Is Included

- FastAPI application factory with thin route modules
- SQLAlchemy 2.0 models and PostgreSQL-first configuration
- Alembic migration scaffolding with an initial schema revision
- Repository layer for persistence concerns
- Service layer for parsing, claim extraction, linking, verification, and audit reporting
- Basic tests for health, parsing, claim extraction, and verification heuristics

## Architecture

The codebase follows a narrow V1 workflow:

1. `Matter` is the top-level workspace.
2. `SourceDocument` stores deposition or evidentiary source metadata.
3. `Segment` stores anchored transcript slices.
4. `Draft` stores drafting artifacts for a matter.
5. `Assertion` stores sentence-level or paragraph-level draft assertions.
6. `ClaimUnit` stores extracted claim fragments from an assertion.
7. `SupportLink` connects a claim unit to one or more supporting segments.
8. `VerificationRun` stores immutable verification outcomes.
9. `UserDecision` stores human override decisions on a verification run.

Routes stay thin. Business rules live in `app/services`. Database access stays in `app/repositories`.

## Project Layout

```text
.
├── alembic/
├── app/
│   ├── api/
│   ├── core/
│   ├── models/
│   ├── repositories/
│   ├── schemas/
│   ├── services/
│   ├── tests/
│   ├── config.py
│   ├── db.py
│   └── main.py
├── .env.example
├── alembic.ini
├── Makefile
├── pyproject.toml
└── README.md
```

## Setup

### 1. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Update `DATABASE_URL` to point at a PostgreSQL database.

### 3. Run migrations

```bash
alembic upgrade head
```

### 4. Start the API

```bash
uvicorn app.main:app --reload
```

The health endpoint is available at [http://localhost:8000/health](http://localhost:8000/health).

## Makefile Shortcuts

- `make install`
- `make dev`
- `make migrate`
- `make test`

## Running Tests

```bash
pytest
```

The included tests focus on service-level heuristics and the health endpoint. They use lightweight in-process execution and do not require PostgreSQL.

## Current V1 Scope

Fully working in this scaffold:

- FastAPI app bootstrap and endpoint wiring
- PostgreSQL-oriented schema and Alembic setup
- Deposition text parsing with page/line anchor extraction when present
- Lightweight claim extraction heuristics
- Deterministic verification heuristics with immutable run persistence and user decisions

Intentionally stubbed or conservative for V1:

- Real PDF parsing / OCR
- Rich exhibit ingestion workflows
- LLM provider API calls
- Semantic reranking beyond simple lexical overlap and rule checks
- Advanced review queues, bundles, and export surfaces

TODO markers in the services highlight the next realistic expansion points without burying the current code in speculative abstractions.

