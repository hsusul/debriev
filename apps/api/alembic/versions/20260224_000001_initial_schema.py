"""Initial schema

Revision ID: 20260224_000001
Revises:
Create Date: 2026-02-24 00:00:01

"""

from __future__ import annotations

from typing import Sequence

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "20260224_000001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "project",
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("project_id"),
    )
    op.create_index(
        op.f("ix_project_project_id"), "project", ["project_id"], unique=False
    )

    op.create_table(
        "document",
        sa.Column("doc_id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=True),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("file_path", sa.String(), nullable=False),
        sa.Column("stub_text", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["project.project_id"]),
        sa.PrimaryKeyConstraint("doc_id"),
    )
    op.create_index(op.f("ix_document_doc_id"), "document", ["doc_id"], unique=False)
    op.create_index(
        op.f("ix_document_project_id"), "document", ["project_id"], unique=False
    )

    op.create_table(
        "report",
        sa.Column("doc_id", sa.Uuid(), nullable=False),
        sa.Column("report_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["doc_id"], ["document.doc_id"]),
        sa.PrimaryKeyConstraint("doc_id"),
    )

    op.create_table(
        "citation",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("doc_id", sa.Uuid(), nullable=False),
        sa.Column("raw", sa.String(), nullable=False),
        sa.Column("normalized", sa.String(), nullable=True),
        sa.Column("start", sa.Integer(), nullable=True),
        sa.Column("end", sa.Integer(), nullable=True),
        sa.Column("context_text", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["doc_id"], ["document.doc_id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_citation_doc_id"), "citation", ["doc_id"], unique=False)

    op.create_table(
        "citationverification",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("input_hash", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("doc_id", sa.String(), nullable=True),
        sa.Column("chunk_id", sa.String(), nullable=True),
        sa.Column("raw_json", sa.Text(), nullable=False),
        sa.Column("summary_status", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_citationverification_input_hash"),
        "citationverification",
        ["input_hash"],
        unique=True,
    )

    op.create_table(
        "verificationresult",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("doc_id", sa.String(), nullable=False),
        sa.Column("input_hash", sa.String(), nullable=False),
        sa.Column("citations_hash", sa.String(), nullable=False),
        sa.Column("citations_json", sa.Text(), nullable=False),
        sa.Column("findings_json", sa.Text(), nullable=False),
        sa.Column("summary_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_verificationresult_doc_id"),
        "verificationresult",
        ["doc_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_verificationresult_input_hash"),
        "verificationresult",
        ["input_hash"],
        unique=False,
    )
    op.create_index(
        op.f("ix_verificationresult_citations_hash"),
        "verificationresult",
        ["citations_hash"],
        unique=False,
    )

    op.create_table(
        "extractedcitations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("doc_id", sa.String(), nullable=False),
        sa.Column("citations_json", sa.Text(), nullable=False),
        sa.Column("evidence_json", sa.Text(), nullable=False),
        sa.Column("probable_case_name_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_extractedcitations_doc_id"),
        "extractedcitations",
        ["doc_id"],
        unique=True,
    )

    op.create_table(
        "verificationjob",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("doc_id", sa.String(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("input_text", sa.Text(), nullable=True),
        sa.Column("error_text", sa.Text(), nullable=True),
        sa.Column("result_id", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_verificationjob_id"),
        "verificationjob",
        ["id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_verificationjob_doc_id"),
        "verificationjob",
        ["doc_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_verificationjob_result_id"),
        "verificationjob",
        ["result_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_verificationjob_result_id"), table_name="verificationjob")
    op.drop_index(op.f("ix_verificationjob_doc_id"), table_name="verificationjob")
    op.drop_index(op.f("ix_verificationjob_id"), table_name="verificationjob")
    op.drop_table("verificationjob")

    op.drop_index(op.f("ix_extractedcitations_doc_id"), table_name="extractedcitations")
    op.drop_table("extractedcitations")

    op.drop_index(
        op.f("ix_verificationresult_citations_hash"),
        table_name="verificationresult",
    )
    op.drop_index(
        op.f("ix_verificationresult_input_hash"), table_name="verificationresult"
    )
    op.drop_index(op.f("ix_verificationresult_doc_id"), table_name="verificationresult")
    op.drop_table("verificationresult")

    op.drop_index(
        op.f("ix_citationverification_input_hash"),
        table_name="citationverification",
    )
    op.drop_table("citationverification")

    op.drop_index(op.f("ix_citation_doc_id"), table_name="citation")
    op.drop_table("citation")

    op.drop_table("report")

    op.drop_index(op.f("ix_document_project_id"), table_name="document")
    op.drop_index(op.f("ix_document_doc_id"), table_name="document")
    op.drop_table("document")

    op.drop_index(op.f("ix_project_project_id"), table_name="project")
    op.drop_table("project")
