"""add persisted draft review runs

Revision ID: 20260406_0010
Revises: 20260404_0009
Create Date: 2026-04-06 00:10:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260406_0010"
down_revision = "20260404_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "draft_review_runs",
        sa.Column("draft_id", sa.Uuid(), nullable=False),
        sa.Column(
            "status",
            sa.Enum("COMPLETED", "FAILED", name="draft_review_run_status_enum"),
            nullable=False,
        ),
        sa.Column("total_claims", sa.Integer(), nullable=False),
        sa.Column("total_flagged_claims", sa.Integer(), nullable=False),
        sa.Column("resolved_flagged_claims", sa.Integer(), nullable=False),
        sa.Column("remaining_flagged_claims", sa.Integer(), nullable=False),
        sa.Column("highest_severity_bucket", sa.String(length=32), nullable=True),
        sa.Column("snapshot_version", sa.Integer(), nullable=False),
        sa.Column("snapshot", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(["draft_id"], ["drafts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_draft_review_runs_draft_id"), "draft_review_runs", ["draft_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_draft_review_runs_draft_id"), table_name="draft_review_runs")
    op.drop_table("draft_review_runs")
    sa.Enum(name="draft_review_run_status_enum").drop(op.get_bind(), checkfirst=True)
