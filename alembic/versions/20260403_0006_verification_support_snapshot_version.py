"""add verification support snapshot version

Revision ID: 20260403_0006
Revises: 20260403_0005
Create Date: 2026-04-03 00:06:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260403_0006"
down_revision = "20260403_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("verification_runs", sa.Column("support_snapshot_version", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("verification_runs", "support_snapshot_version")
