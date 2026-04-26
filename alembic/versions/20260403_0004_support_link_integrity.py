"""add support link uniqueness constraint

Revision ID: 20260403_0004
Revises: 20260403_0003
Create Date: 2026-04-03 00:04:00
"""

from alembic import op


revision = "20260403_0004"
down_revision = "20260403_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("support_links") as batch_op:
        batch_op.create_unique_constraint(
            "uq_support_links_claim_unit_segment",
            ["claim_unit_id", "segment_id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("support_links") as batch_op:
        batch_op.drop_constraint(
            "uq_support_links_claim_unit_segment",
            type_="unique",
        )
