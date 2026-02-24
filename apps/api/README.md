# Debriev API

## Migrations (Alembic)

Run from `apps/api`:

```bash
alembic upgrade head
```

Create a new migration:

```bash
alembic revision -m "describe change"
```

Autogenerate from SQLModel metadata:

```bash
alembic revision --autogenerate -m "describe change"
```

### Dev Auto-Migrate

Set `DEV_AUTO_MIGRATE=true` to run `alembic upgrade head` on API startup.
Default is `false`.

## Verification Worker

Run the durable verification job worker:

```bash
python -m app.worker
```

Process at most one queued job and exit:

```bash
python -m app.worker --once
```
