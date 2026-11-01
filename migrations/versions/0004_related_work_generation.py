"""Persist related-work generations and durable background jobs."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(op.f("ck_model_versions_role"), "model_versions", type_="check")
    op.create_check_constraint(
        op.f("ck_model_versions_role"),
        "model_versions",
        "role IN ('paper_embedding','chunk_embedding','reranking','extraction','summarization','generation')",
    )
    op.create_table(
        "related_work_generations",
        sa.Column("paper_id", sa.Uuid(), nullable=False),
        sa.Column("model_key", sa.String(24), nullable=False),
        sa.Column("model_version_id", sa.Uuid(), nullable=True),
        sa.Column("model_repository_id", sa.Text(), nullable=False),
        sa.Column("model_revision", sa.Text(), nullable=False),
        sa.Column("prompt_version", sa.Text(), nullable=False),
        sa.Column("generation_version", sa.Text(), nullable=False),
        sa.Column("generation_config", postgresql.JSONB(), nullable=False),
        sa.Column("cache_key", sa.CHAR(64), nullable=False),
        sa.Column("evidence_hash", sa.CHAR(64), nullable=False),
        sa.Column("input_hash", sa.CHAR(64), nullable=False),
        sa.Column("evidence_snapshot", postgresql.JSONB(), nullable=False),
        sa.Column("generated_text", sa.Text(), nullable=True),
        sa.Column(
            "citations",
            postgresql.ARRAY(sa.Text()),
            server_default=sa.text("'{}'::text[]"),
            nullable=False,
        ),
        sa.Column(
            "source_references",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("input_token_count", sa.Integer(), nullable=True),
        sa.Column("output_token_count", sa.Integer(), nullable=True),
        sa.Column("context_limit", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "model_key IN ('primera','gemma')", name="ck_related_work_generations_model_key"
        ),
        sa.CheckConstraint(
            "status IN ('completed','context_overflow','failed')",
            name="ck_related_work_generations_status",
        ),
        sa.CheckConstraint(
            "cache_key ~ '^[0-9a-f]{64}$'", name="ck_related_work_generations_cache_key"
        ),
        sa.CheckConstraint(
            "evidence_hash ~ '^[0-9a-f]{64}$'", name="ck_related_work_generations_evidence_hash"
        ),
        sa.CheckConstraint(
            "input_hash ~ '^[0-9a-f]{64}$'", name="ck_related_work_generations_input_hash"
        ),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"], ondelete="RESTRICT"),
    )
    op.create_index(
        "ix_related_work_generations_paper_id", "related_work_generations", ["paper_id"]
    )
    op.create_index(
        "ix_related_work_generations_model_key", "related_work_generations", ["model_key"]
    )
    op.create_index(
        "ix_related_work_generations_model_version_id",
        "related_work_generations",
        ["model_version_id"],
    )
    op.create_index(
        "ix_related_work_generations_cache_key", "related_work_generations", ["cache_key"]
    )
    op.create_index(
        "ix_related_work_generations_status", "related_work_generations", ["status"]
    )
    op.create_table(
        "related_work_jobs",
        sa.Column("paper_id", sa.Uuid(), nullable=False),
        sa.Column("model_key", sa.String(24), nullable=False),
        sa.Column("status", sa.String(16), server_default="pending", nullable=False),
        sa.Column(
            "force_regenerate", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column("cache_key", sa.CHAR(64), nullable=False),
        sa.Column("generation_id", sa.Uuid(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "model_key IN ('primera','gemma')", name="ck_related_work_jobs_model_key"
        ),
        sa.CheckConstraint(
            "status IN ('pending','running','completed','failed')",
            name="ck_related_work_jobs_status",
        ),
        sa.CheckConstraint("cache_key ~ '^[0-9a-f]{64}$'", name="ck_related_work_jobs_cache_key"),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["generation_id"], ["related_work_generations.id"], ondelete="SET NULL"
        ),
    )
    op.create_index("ix_related_work_jobs_paper_id", "related_work_jobs", ["paper_id"])
    op.create_index("ix_related_work_jobs_model_key", "related_work_jobs", ["model_key"])
    op.create_index("ix_related_work_jobs_status", "related_work_jobs", ["status"])
    op.create_index("ix_related_work_jobs_generation_id", "related_work_jobs", ["generation_id"])
    op.create_index(
        "uq_related_work_jobs_active_model",
        "related_work_jobs",
        ["paper_id", "model_key"],
        unique=True,
        postgresql_where=sa.text("status IN ('pending','running')"),
    )


def downgrade() -> None:
    op.drop_table("related_work_jobs")
    op.drop_table("related_work_generations")
    op.execute("DELETE FROM model_versions WHERE role = 'generation'")
    op.drop_constraint(op.f("ck_model_versions_role"), "model_versions", type_="check")
    op.create_check_constraint(
        op.f("ck_model_versions_role"),
        "model_versions",
        "role IN ('paper_embedding','chunk_embedding','reranking','extraction','summarization')",
    )
