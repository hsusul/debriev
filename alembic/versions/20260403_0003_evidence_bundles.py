"""add evidence bundles

Revision ID: 20260403_0003
Revises: 20260403_0002
Create Date: 2026-04-03 00:03:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260403_0003"
down_revision = "20260403_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "evidence_bundles",
        sa.Column("matter_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["matter_id"], ["matters.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_evidence_bundles_matter_id", "evidence_bundles", ["matter_id"])

    op.create_table(
        "evidence_bundle_source_documents",
        sa.Column("evidence_bundle_id", sa.Uuid(), nullable=False),
        sa.Column("source_document_id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["evidence_bundle_id"], ["evidence_bundles.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_document_id"], ["source_documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("evidence_bundle_id", "source_document_id"),
    )

    op.add_column("drafts", sa.Column("evidence_bundle_id", sa.Uuid(), nullable=True))
    op.create_index("ix_drafts_evidence_bundle_id", "drafts", ["evidence_bundle_id"])
    op.create_foreign_key(
        "fk_drafts_evidence_bundle_id_evidence_bundles",
        "drafts",
        "evidence_bundles",
        ["evidence_bundle_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_drafts_evidence_bundle_id_evidence_bundles", "drafts", type_="foreignkey")
    op.drop_index("ix_drafts_evidence_bundle_id", table_name="drafts")
    op.drop_column("drafts", "evidence_bundle_id")

    op.drop_table("evidence_bundle_source_documents")

    op.drop_index("ix_evidence_bundles_matter_id", table_name="evidence_bundles")
    op.drop_table("evidence_bundles")
