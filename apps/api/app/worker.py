from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import time
from typing import Any

try:
    from sqlmodel import Session, select
except ModuleNotFoundError:  # pragma: no cover - requires sqlmodel runtime
    Session = Any  # type: ignore[assignment]

    def select(*args: Any, **kwargs: Any) -> Any:
        raise ModuleNotFoundError("sqlmodel is required for worker execution")


from app.db import VerificationJob
from app.db.session import engine
from app.main import _execute_verification_job
from app.settings import settings


def _is_stale(updated_at: datetime, *, stale_minutes: int) -> bool:
    cutoff = datetime.now(UTC) - timedelta(minutes=max(1, stale_minutes))
    if updated_at.tzinfo is None:
        return updated_at < cutoff.replace(tzinfo=None)
    return updated_at < cutoff


def requeue_stale_running_jobs(
    session: Session,
    *,
    stale_minutes: int = 15,
) -> int:
    jobs = session.exec(
        select(VerificationJob).where(VerificationJob.status == "running")
    ).all()
    requeued = 0
    now = datetime.now(UTC)
    for job in jobs:
        if not _is_stale(job.updated_at, stale_minutes=stale_minutes):
            continue
        if job.result_id is not None:
            job.status = "done"
        else:
            job.status = "queued"
            job.error_text = (
                f"Requeued stale running job after {max(1, stale_minutes)} minutes"
            )
        job.updated_at = now
        session.add(job)
        requeued += 1
    if requeued:
        session.commit()
    return requeued


def claim_next_queued_job(session: Session) -> VerificationJob | None:
    queued = session.exec(
        select(VerificationJob)
        .where(VerificationJob.status == "queued")
        .order_by(VerificationJob.created_at.asc(), VerificationJob.id.asc())
    ).first()
    if queued is None:
        return None

    if queued.result_id is not None:
        queued.status = "done"
        queued.updated_at = datetime.now(UTC)
        session.add(queued)
        session.commit()
        return None

    queued.status = "running"
    queued.error_text = None
    queued.updated_at = datetime.now(UTC)
    session.add(queued)
    session.commit()
    session.refresh(queued)
    return queued


def run_worker_iteration(
    session: Session,
    *,
    stale_minutes: int = 15,
) -> str:
    requeue_stale_running_jobs(session, stale_minutes=stale_minutes)
    job = claim_next_queued_job(session)
    if job is None:
        return "idle"

    _execute_verification_job(job.id, job.input_text, session)
    return job.id


def run_worker_loop(
    *,
    poll_seconds: float,
    stale_minutes: int,
    once: bool,
) -> None:
    safe_poll = max(0.1, poll_seconds)
    while True:
        with Session(engine) as session:
            result = run_worker_iteration(session, stale_minutes=stale_minutes)

        if once:
            return

        if result == "idle":
            time.sleep(safe_poll)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debriev verification job worker")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process at most one queued job and exit",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=settings.verification_worker_poll_seconds,
        help="Polling interval in seconds when idle",
    )
    parser.add_argument(
        "--stale-minutes",
        type=int,
        default=15,
        help="Minutes before a running job is considered stale",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_worker_loop(
        poll_seconds=args.poll_seconds,
        stale_minutes=args.stale_minutes,
        once=args.once,
    )


if __name__ == "__main__":
    main()
