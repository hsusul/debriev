import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.deps import get_db_session
from app.config import get_settings
from app.core.enums import SupportStatus
from app.main import create_app
from app.models import Base


@pytest.fixture
def session() -> Session:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    db_session = SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()
        engine.dispose()


@pytest.fixture
def client(session: Session) -> TestClient:
    app = create_app()

    def override_get_db():
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[get_db_session] = override_get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_case_pdf_verification_supports_statement_against_text_pdf_and_matches_citation(
    client: TestClient,
) -> None:
    pdf_bytes = _build_text_pdf(
        [
            "Celotex Corp. v. Catrett, 477 U.S. 317 (1986)",
            "The burden on the moving party may be discharged by showing that there is an absence of evidence to support the nonmoving party's case.",
        ]
    )

    response = client.post(
        "/api/v1/case-pdf-verification",
        files={"pdf_file": ("celotex.pdf", pdf_bytes, "application/pdf")},
        data={
            "statement_text": "The burden on the moving party may be discharged by showing that there is an absence of evidence to support the nonmoving party's case.",
            "citation_text": "Celotex Corp. v. Catrett, 477 U.S. 317 (1986)",
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["pdf_text_status"] == "text_extracted"
    assert payload["extracted_character_count"] > 100
    assert payload["page_count"] == 1
    assert payload["extracted_text_preview"].startswith("Celotex Corp. v. Catrett")
    assert payload["citation_match_status"] == "citation_matches_pdf"
    assert payload["statement_verdict"] == SupportStatus.SUPPORTED.value
    assert payload["extracted_authority_metadata"]["case_name"] == "Celotex Corp. v. Catrett"
    assert payload["extracted_authority_metadata"]["reporter_volume"] == "477"
    assert payload["support_snippet"] is not None


def test_pdf_citation_verification_extracts_text_and_returns_citation_rows(
    client: TestClient,
) -> None:
    pdf_bytes = _build_text_pdf(
        [
            "Citation Memo",
            "Brown v. Board of Education, 347 U.S. 483 (1954), held that segregation in public schools deprives children of equal educational opportunities.",
            "Anderson v. Liberty Lobby, Inc., 477 U.S. 242 (1986), held that summary judgment always fails when facts are disputed.",
        ]
    )

    response = client.post(
        "/api/v1/citation-verification/pdf",
        files={"pdf_file": ("draft-memo.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["pdf_text_status"] == "text_extracted"
    assert payload["extracted_character_count"] > 100
    assert payload["page_count"] == 1
    assert "Brown v. Board of Education" in payload["extracted_text_preview"]
    citation_verification = payload["citation_verification"]
    assert citation_verification is not None
    assert citation_verification["summary"]["total_cited_propositions"] == len(citation_verification["citations"])
    assert len(citation_verification["citations"]) == 2
    assert citation_verification["citations"][0]["citation_text"] == "Brown v. Board of Education, 347 U.S. 483 (1954)"


def test_pdf_citation_verification_returns_honest_status_for_unreadable_pdf(
    client: TestClient,
) -> None:
    pdf_bytes = _build_pdf_with_stream("q\nQ")

    response = client.post(
        "/api/v1/citation-verification/pdf",
        files={"pdf_file": ("draft-scan.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["pdf_text_status"] == "scanned_or_unreadable"
    assert payload["extracted_character_count"] == 0
    assert payload["page_count"] == 1
    assert payload["extraction_warnings"]
    assert payload["extracted_text_preview"] is None
    assert payload["citation_verification"] is None


def test_case_pdf_verification_returns_unsupported_for_statement_not_supported_by_pdf_text(
    client: TestClient,
) -> None:
    pdf_bytes = _build_text_pdf(
        [
            "Brown v. Board of Education, 347 U.S. 483 (1954)",
            "Segregation of children in public schools solely on the basis of race deprives minority children of equal educational opportunities.",
        ]
    )

    response = client.post(
        "/api/v1/case-pdf-verification",
        files={"pdf_file": ("brown.pdf", pdf_bytes, "application/pdf")},
        data={"statement_text": "Brown held that negligence is always enough."},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["pdf_text_status"] == "text_extracted"
    assert payload["extracted_character_count"] > 0
    assert payload["page_count"] == 1
    assert payload["citation_match_status"] == "citation_not_provided"
    assert payload["statement_verdict"] == SupportStatus.UNSUPPORTED.value
    assert payload["reasoning"] is not None


def test_case_pdf_verification_returns_citation_mismatch_when_citation_conflicts_with_pdf(
    client: TestClient,
) -> None:
    pdf_bytes = _build_text_pdf(
        [
            "Celotex Corp. v. Catrett, 477 U.S. 317 (1986)",
            "The burden on the moving party may be discharged by showing that there is an absence of evidence to support the nonmoving party's case.",
        ]
    )

    response = client.post(
        "/api/v1/case-pdf-verification",
        files={"pdf_file": ("celotex.pdf", pdf_bytes, "application/pdf")},
        data={
            "statement_text": "The burden on the moving party may be discharged by showing that there is an absence of evidence to support the nonmoving party's case.",
            "citation_text": "Brown v. Board of Education, 347 U.S. 483 (1954)",
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["citation_match_status"] == "citation_mismatch"
    assert payload["statement_verdict"] == SupportStatus.SUPPORTED.value


def test_case_pdf_verification_returns_citation_recognized_when_pdf_metadata_is_missing(
    client: TestClient,
) -> None:
    pdf_bytes = _build_text_pdf(
        [
            "This opinion explains the applicable standard.",
            "The burden on the moving party may be discharged by showing that there is an absence of evidence to support the nonmoving party's case.",
        ]
    )

    response = client.post(
        "/api/v1/case-pdf-verification",
        files={"pdf_file": ("unknown.pdf", pdf_bytes, "application/pdf")},
        data={
            "statement_text": "The burden on the moving party may be discharged by showing that there is an absence of evidence to support the nonmoving party's case.",
            "citation_text": "Celotex Corp. v. Catrett, 477 U.S. 317 (1986)",
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["citation_match_status"] == "citation_recognized"
    assert payload["extracted_authority_metadata"] is None


def test_case_pdf_verification_returns_scanned_or_unreadable_when_no_text_can_be_extracted(
    client: TestClient,
) -> None:
    pdf_bytes = _build_pdf_with_stream("q\nQ")

    response = client.post(
        "/api/v1/case-pdf-verification",
        files={"pdf_file": ("scanned.pdf", pdf_bytes, "application/pdf")},
        data={"statement_text": "Any statement."},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["pdf_text_status"] == "scanned_or_unreadable"
    assert payload["extracted_character_count"] == 0
    assert payload["page_count"] == 1
    assert payload["extracted_text_preview"] is None
    assert payload["extraction_warnings"]
    assert payload["statement_verdict"] == SupportStatus.UNVERIFIED.value
    assert payload["support_snippet"] is None


def _build_text_pdf(lines: list[str]) -> bytes:
    commands = ["BT", "/F1 12 Tf", "72 720 Td"]
    for index, line in enumerate(lines):
        if index > 0:
            commands.append("0 -18 Td")
        commands.append(f"({_escape_pdf_text(line)}) Tj")
    commands.append("ET")
    return _build_pdf_with_stream("\n".join(commands))


def _build_pdf_with_stream(stream_text: str) -> bytes:
    stream_bytes = stream_text.encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("latin-1") + stream_bytes + b"\nendstream",
    ]

    parts: list[bytes] = [b"%PDF-1.4\n"]
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(sum(len(part) for part in parts))
        parts.append(f"{index} 0 obj\n".encode("latin-1") + obj + b"\nendobj\n")

    xref_offset = sum(len(part) for part in parts)
    xref_lines = [b"xref\n", f"0 {len(objects) + 1}\n".encode("latin-1"), b"0000000000 65535 f \n"]
    for offset in offsets[1:]:
        xref_lines.append(f"{offset:010d} 00000 n \n".encode("latin-1"))

    trailer = (
        b"trailer\n"
        + f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode("latin-1")
        + b"startxref\n"
        + f"{xref_offset}\n".encode("latin-1")
        + b"%%EOF"
    )
    return b"".join(parts + xref_lines + [trailer])


def _escape_pdf_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
