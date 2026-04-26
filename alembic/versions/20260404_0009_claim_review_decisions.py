"""add claim review decisions

Revision ID: 20260404_0009
Revises: 20260403_0008
Create Date: 2026-04-04 00:09:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260404_0009"
down_revision = "20260403_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "claim_review_decisions",
        sa.Column("claim_unit_id", sa.Uuid(), nullable=False),
        sa.Column("draft_id", sa.Uuid(), nullable=False),
        sa.Column("verification_run_id", sa.Uuid(), nullable=True),
        sa.Column(
            "action",
            sa.Enum(
                "acknowledge_risk",
                "mark_for_revision",
                "resolve_with_edit",
                name="claim_review_action_enum",
            ),
            nullable=False,
        ),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("proposed_replacement_text", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(["claim_unit_id"], ["claim_units.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["draft_id"], ["drafts.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["verification_run_id"], ["verification_runs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_claim_review_decisions_claim_unit_id"),
        "claim_review_decisions",
        ["claim_unit_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_claim_review_decisions_draft_id"),
        "claim_review_decisions",
        ["draft_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_claim_review_decisions_verification_run_id"),
        "claim_review_decisions",
        ["verification_run_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_claim_review_decisions_verification_run_id"), table_name="claim_review_decisions")
    op.drop_index(op.f("ix_claim_review_decisions_draft_id"), table_name="claim_review_decisions")
    op.drop_index(op.f("ix_claim_review_decisions_claim_unit_id"), table_name="claim_review_decisions")
    op.drop_table("claim_review_decisions")
    sa.Enum(name="claim_review_action_enum").drop(op.get_bind(), checkfirst=True)
