from __future__ import annotations

from uuid import uuid4
from unittest.mock import patch

from sqlmodel import SQLModel, Session, create_engine, select

from app.courtlistener import CourtListenerError
from app.db import Document, VerificationJob, VerificationResult
from app.worker import run_worker_iteration


def _make_session(tmp_path):
    db_path = tmp_path / "worker-test.db"
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def _insert_document(session: Session) -> str:
    doc_id = uuid4()
    session.add(
        Document(
            doc_id=doc_id,
            filename="worker.pdf",
            file_path="/tmp/worker.pdf",
            stub_text="",
        )
    )
    session.commit()
    return str(doc_id)


def test_worker_processes_queued_job_and_stores_result(tmp_path) -> None:
    with _make_session(tmp_path) as session:
        doc_id = _insert_document(session)
        job = VerificationJob(
            doc_id=doc_id,
            status="queued",
            input_text="Roe v. Wade, 410 U.S. 113 (1973).",
        )
        session.add(job)
        session.commit()
        session.refresh(job)

        with patch(
            "app.main.CourtListenerClient.lookup_citation_list",
            return_value={
                "results": [
                    {
                        "citation": "410 U.S. 113",
                        "results": [{"citation": "410 U.S. 113"}],
                    }
                ]
            },
        ):
            processed = run_worker_iteration(session)

        stored_job = session.exec(
            select(VerificationJob).where(VerificationJob.id == job.id)
        ).first()
        stored_result = session.exec(
            select(VerificationResult).where(VerificationResult.doc_id == doc_id)
        ).first()

    assert processed == job.id
    assert stored_job is not None
    assert stored_job.status == "done"
    assert stored_job.result_id is not None
    assert stored_result is not None
    assert stored_result.id == stored_job.result_id


def test_worker_failure_sets_failed_status(tmp_path) -> None:
    with _make_session(tmp_path) as session:
        doc_id = _insert_document(session)
        job = VerificationJob(
            doc_id=doc_id,
            status="queued",
            input_text="Roe v. Wade, 410 U.S. 113 (1973).",
        )
        session.add(job)
        session.commit()
        session.refresh(job)

        with patch(
            "app.main.CourtListenerClient.lookup_citation_list",
            side_effect=CourtListenerError("worker forced failure"),
        ):
            processed = run_worker_iteration(session)

        stored_job = session.exec(
            select(VerificationJob).where(VerificationJob.id == job.id)
        ).first()

    assert processed == job.id
    assert stored_job is not None
    assert stored_job.status == "failed"
    assert stored_job.result_id is None
    assert stored_job.error_text is not None
    assert "worker forced failure" in stored_job.error_text
