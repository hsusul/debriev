"""add assertion extraction versioning and snapshot

Revision ID: 20260403_0007
Revises: 20260403_0006
Create Date: 2026-04-03 00:07:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260403_0007"
down_revision = "20260403_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("assertions", sa.Column("extraction_strategy", sa.String(length=32), nullable=True))
    op.add_column("assertions", sa.Column("extraction_version", sa.Integer(), nullable=True))
    op.add_column("assertions", sa.Column("extraction_snapshot", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("assertions", "extraction_snapshot")
    op.drop_column("assertions", "extraction_version")
    op.drop_column("assertions", "extraction_strategy")
