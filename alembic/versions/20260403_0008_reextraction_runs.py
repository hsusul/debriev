"""add persisted re-extraction runs

Revision ID: 20260403_0008
Revises: 20260403_0007
Create Date: 2026-04-03 00:08:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260403_0008"
down_revision = "20260403_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "reextraction_runs",
        sa.Column("draft_id", sa.Uuid(), nullable=False),
        sa.Column(
            "run_kind",
            sa.Enum("PREVIEW", "APPLY", name="reextraction_run_kind_enum"),
            nullable=False,
        ),
        sa.Column("requested_mode", sa.String(length=16), nullable=False),
        sa.Column("extraction_version", sa.Integer(), nullable=False),
        sa.Column("total_assertions", sa.Integer(), nullable=False),
        sa.Column("ready_assertions", sa.Integer(), nullable=False),
        sa.Column("unchanged_assertions", sa.Integer(), nullable=False),
        sa.Column("applied_assertions", sa.Integer(), nullable=False),
        sa.Column("skipped_assertions", sa.Integer(), nullable=False),
        sa.Column("blocked_assertions", sa.Integer(), nullable=False),
        sa.Column("materially_changed_assertions", sa.Integer(), nullable=False),
        sa.Column("legacy_unversioned_assertions", sa.Integer(), nullable=False),
        sa.Column("replaced_assertions", sa.Integer(), nullable=False),
        sa.Column("metadata_only_assertions", sa.Integer(), nullable=False),
        sa.Column("snapshot_version", sa.Integer(), nullable=False),
        sa.Column("snapshot", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(["draft_id"], ["drafts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_reextraction_runs_draft_id"), "reextraction_runs", ["draft_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_reextraction_runs_draft_id"), table_name="reextraction_runs")
    op.drop_table("reextraction_runs")
    sa.Enum(name="reextraction_run_kind_enum").drop(op.get_bind(), checkfirst=True)
