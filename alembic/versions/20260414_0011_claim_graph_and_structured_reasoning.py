"""add claim graph edges and structured reasoning categories

Revision ID: 20260414_0011
Revises: 20260406_0010
Create Date: 2026-04-14 00:11:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260414_0011"
down_revision = "20260406_0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "verification_runs",
        sa.Column("reasoning_categories", sa.JSON(), nullable=False, server_default="[]"),
    )

    op.create_table(
        "claim_graph_edges",
        sa.Column("draft_id", sa.Uuid(), nullable=False),
        sa.Column("draft_review_run_id", sa.Uuid(), nullable=False),
        sa.Column("source_claim_id", sa.Uuid(), nullable=False),
        sa.Column("target_claim_id", sa.Uuid(), nullable=False),
        sa.Column(
            "relationship_type",
            sa.Enum(
                "SUPPORTS",
                "CONTRADICTS",
                "DEPENDS_ON",
                "DUPLICATE_OF",
                name="claim_graph_relationship_type_enum",
            ),
            nullable=False,
        ),
        sa.Column("reason_code", sa.String(length=64), nullable=True),
        sa.Column("reason_text", sa.Text(), nullable=True),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(["draft_id"], ["drafts.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["draft_review_run_id"], ["draft_review_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_claim_id"], ["claim_units.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["target_claim_id"], ["claim_units.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "draft_review_run_id",
            "source_claim_id",
            "target_claim_id",
            "relationship_type",
            name="uq_claim_graph_edge_run_source_target_type",
        ),
    )
    op.create_index(op.f("ix_claim_graph_edges_draft_id"), "claim_graph_edges", ["draft_id"], unique=False)
    op.create_index(
        op.f("ix_claim_graph_edges_draft_review_run_id"),
        "claim_graph_edges",
        ["draft_review_run_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_claim_graph_edges_source_claim_id"),
        "claim_graph_edges",
        ["source_claim_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_claim_graph_edges_target_claim_id"),
        "claim_graph_edges",
        ["target_claim_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_claim_graph_edges_target_claim_id"), table_name="claim_graph_edges")
    op.drop_index(op.f("ix_claim_graph_edges_source_claim_id"), table_name="claim_graph_edges")
    op.drop_index(op.f("ix_claim_graph_edges_draft_review_run_id"), table_name="claim_graph_edges")
    op.drop_index(op.f("ix_claim_graph_edges_draft_id"), table_name="claim_graph_edges")
    op.drop_table("claim_graph_edges")
    sa.Enum(name="claim_graph_relationship_type_enum").drop(op.get_bind(), checkfirst=True)

    op.drop_column("verification_runs", "reasoning_categories")
