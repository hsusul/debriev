"""add verification support snapshot

Revision ID: 20260403_0005
Revises: 20260403_0004
Create Date: 2026-04-03 00:05:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260403_0005"
down_revision = "20260403_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("verification_runs", sa.Column("support_snapshot", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("verification_runs", "support_snapshot")
