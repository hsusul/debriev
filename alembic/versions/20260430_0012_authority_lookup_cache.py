"""add authority lookup cache results

Revision ID: 20260430_0012
Revises: 20260414_0011
Create Date: 2026-04-30 00:12:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260430_0012"
down_revision = "20260414_0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "authority_lookup_cache_results",
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("lookup_key", sa.String(length=512), nullable=False),
        sa.Column("normalized_resource_key", sa.String(length=512), nullable=True),
        sa.Column("volume", sa.String(length=32), nullable=True),
        sa.Column("reporter", sa.String(length=64), nullable=True),
        sa.Column("page", sa.String(length=32), nullable=True),
        sa.Column("case_name", sa.String(length=512), nullable=True),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("lookup_status", sa.String(length=64), nullable=False),
        sa.Column("matched_provider_cluster_id", sa.String(length=128), nullable=True),
        sa.Column("matched_case_name", sa.String(length=512), nullable=True),
        sa.Column("matched_canonical_citation", sa.String(length=512), nullable=True),
        sa.Column("matched_absolute_url", sa.String(length=1024), nullable=True),
        sa.Column("matched_date_filed", sa.String(length=32), nullable=True),
        sa.Column("matched_year", sa.Integer(), nullable=True),
        sa.Column("normalized_citations", sa.JSON(), nullable=False),
        sa.Column("raw_lookup_payload", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("looked_up_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("provider", "lookup_key", name="uq_authority_lookup_cache_provider_lookup_key"),
    )
    op.create_index(
        op.f("ix_authority_lookup_cache_results_provider"),
        "authority_lookup_cache_results",
        ["provider"],
        unique=False,
    )
    op.create_index(
        op.f("ix_authority_lookup_cache_results_lookup_key"),
        "authority_lookup_cache_results",
        ["lookup_key"],
        unique=False,
    )
    op.create_index(
        op.f("ix_authority_lookup_cache_results_normalized_resource_key"),
        "authority_lookup_cache_results",
        ["normalized_resource_key"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_authority_lookup_cache_results_normalized_resource_key"), table_name="authority_lookup_cache_results")
    op.drop_index(op.f("ix_authority_lookup_cache_results_lookup_key"), table_name="authority_lookup_cache_results")
    op.drop_index(op.f("ix_authority_lookup_cache_results_provider"), table_name="authority_lookup_cache_results")
    op.drop_table("authority_lookup_cache_results")
