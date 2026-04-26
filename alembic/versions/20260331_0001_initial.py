"""initial schema

Revision ID: 20260331_0001
Revises:
Create Date: 2026-03-31 00:01:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260331_0001"
down_revision = None
branch_labels = None
depends_on = None


source_type_enum = sa.Enum("DEPOSITION", "EXHIBIT", "DECLARATION", name="source_type_enum")
parser_status_enum = sa.Enum("PENDING", "PROCESSING", "COMPLETED", "FAILED", name="parser_status_enum")
draft_mode_enum = sa.Enum("DRAFT", "COMPILE", "AUDIT", name="draft_mode_enum")
claim_type_enum = sa.Enum("FACT", "INFERENCE", "QUOTE", "MIXED", name="claim_type_enum")
support_status_enum = sa.Enum(
    "UNVERIFIED",
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "OVERSTATED",
    "UNSUPPORTED",
    "CONTRADICTED",
    "AMBIGUOUS",
    name="support_status_enum",
)
link_type_enum = sa.Enum("MANUAL", "AUTO_SUGGESTED", "AUTO_ACCEPTED", name="link_type_enum")
verification_verdict_enum = sa.Enum(
    "UNVERIFIED",
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "OVERSTATED",
    "UNSUPPORTED",
    "CONTRADICTED",
    "AMBIGUOUS",
    name="verification_verdict_enum",
)
decision_action_enum = sa.Enum(
    "ACKNOWLEDGE_INFERENCE",
    "INTENTIONAL_ADVOCACY",
    "NEEDS_CITATION_LATER",
    "FALSE_POSITIVE",
    "ESCALATE_FOR_REVIEW",
    name="decision_action_enum",
)


def upgrade() -> None:
    bind = op.get_bind()
    source_type_enum.create(bind, checkfirst=True)
    parser_status_enum.create(bind, checkfirst=True)
    draft_mode_enum.create(bind, checkfirst=True)
    claim_type_enum.create(bind, checkfirst=True)
    support_status_enum.create(bind, checkfirst=True)
    link_type_enum.create(bind, checkfirst=True)
    verification_verdict_enum.create(bind, checkfirst=True)
    decision_action_enum.create(bind, checkfirst=True)

    op.create_table(
        "matters",
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("court", sa.String(length=255), nullable=True),
        sa.Column("jurisdiction", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "source_documents",
        sa.Column("matter_id", sa.Uuid(), nullable=False),
        sa.Column("file_name", sa.String(length=255), nullable=False),
        sa.Column("source_type", source_type_enum, nullable=False),
        sa.Column("raw_file_path", sa.Text(), nullable=False),
        sa.Column("parser_status", parser_status_enum, nullable=False),
        sa.Column("parser_confidence", sa.Float(), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["matter_id"], ["matters.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_source_documents_matter_id", "source_documents", ["matter_id"])

    op.create_table(
        "drafts",
        sa.Column("matter_id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("mode", draft_mode_enum, nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["matter_id"], ["matters.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_drafts_matter_id", "drafts", ["matter_id"])

    op.create_table(
        "segments",
        sa.Column("source_document_id", sa.Uuid(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("line_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("line_end", sa.Integer(), nullable=True),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("normalized_text", sa.Text(), nullable=False),
        sa.Column("speaker", sa.String(length=100), nullable=True),
        sa.Column("segment_type", sa.String(length=50), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["source_document_id"], ["source_documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_segments_source_document_id", "segments", ["source_document_id"])

    op.create_table(
        "assertions",
        sa.Column("draft_id", sa.Uuid(), nullable=False),
        sa.Column("paragraph_index", sa.Integer(), nullable=True),
        sa.Column("sentence_index", sa.Integer(), nullable=True),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("normalized_text", sa.Text(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["draft_id"], ["drafts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_assertions_draft_id", "assertions", ["draft_id"])

    op.create_table(
        "claim_units",
        sa.Column("assertion_id", sa.Uuid(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("normalized_text", sa.Text(), nullable=False),
        sa.Column("claim_type", claim_type_enum, nullable=False),
        sa.Column("sequence_order", sa.Integer(), nullable=False),
        sa.Column("support_status", support_status_enum, nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["assertion_id"], ["assertions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_claim_units_assertion_id", "claim_units", ["assertion_id"])

    op.create_table(
        "verification_runs",
        sa.Column("claim_unit_id", sa.Uuid(), nullable=False),
        sa.Column("model_version", sa.String(length=100), nullable=False),
        sa.Column("prompt_version", sa.String(length=100), nullable=False),
        sa.Column("deterministic_flags", sa.JSON(), nullable=False),
        sa.Column("verdict", verification_verdict_enum, nullable=False),
        sa.Column("reasoning", sa.Text(), nullable=False),
        sa.Column("suggested_fix", sa.Text(), nullable=True),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["claim_unit_id"], ["claim_units.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_verification_runs_claim_unit_id", "verification_runs", ["claim_unit_id"])

    op.create_table(
        "support_links",
        sa.Column("claim_unit_id", sa.Uuid(), nullable=False),
        sa.Column("segment_id", sa.Uuid(), nullable=False),
        sa.Column("sequence_order", sa.Integer(), nullable=True),
        sa.Column("link_type", link_type_enum, nullable=False),
        sa.Column("citation_text", sa.Text(), nullable=True),
        sa.Column("user_confirmed", sa.Boolean(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["claim_unit_id"], ["claim_units.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["segment_id"], ["segments.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_support_links_claim_unit_id", "support_links", ["claim_unit_id"])
    op.create_index("ix_support_links_segment_id", "support_links", ["segment_id"])

    op.create_table(
        "user_decisions",
        sa.Column("verification_run_id", sa.Uuid(), nullable=False),
        sa.Column("action", decision_action_enum, nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["verification_run_id"], ["verification_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_user_decisions_verification_run_id", "user_decisions", ["verification_run_id"])


def downgrade() -> None:
    op.drop_index("ix_user_decisions_verification_run_id", table_name="user_decisions")
    op.drop_table("user_decisions")
    op.drop_index("ix_support_links_segment_id", table_name="support_links")
    op.drop_index("ix_support_links_claim_unit_id", table_name="support_links")
    op.drop_table("support_links")
    op.drop_index("ix_verification_runs_claim_unit_id", table_name="verification_runs")
    op.drop_table("verification_runs")
    op.drop_index("ix_claim_units_assertion_id", table_name="claim_units")
    op.drop_table("claim_units")
    op.drop_index("ix_assertions_draft_id", table_name="assertions")
    op.drop_table("assertions")
    op.drop_index("ix_segments_source_document_id", table_name="segments")
    op.drop_table("segments")
    op.drop_index("ix_drafts_matter_id", table_name="drafts")
    op.drop_table("drafts")
    op.drop_index("ix_source_documents_matter_id", table_name="source_documents")
    op.drop_table("source_documents")
    op.drop_table("matters")

    bind = op.get_bind()
    decision_action_enum.drop(bind, checkfirst=True)
    verification_verdict_enum.drop(bind, checkfirst=True)
    link_type_enum.drop(bind, checkfirst=True)
    support_status_enum.drop(bind, checkfirst=True)
    claim_type_enum.drop(bind, checkfirst=True)
    draft_mode_enum.drop(bind, checkfirst=True)
    parser_status_enum.drop(bind, checkfirst=True)
    source_type_enum.drop(bind, checkfirst=True)
