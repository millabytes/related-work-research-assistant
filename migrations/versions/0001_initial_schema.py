"""initial_schema

Revision ID: 0001
Revises:
Create Date: 2026-10-31 14:30:35.348395

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import VECTOR
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""

    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "model_versions",
        sa.Column("model_key", sa.Text(), nullable=False),
        sa.Column("role", sa.String(length=30), nullable=False),
        sa.Column("provider", sa.Text(), nullable=True),
        sa.Column("repository_id", sa.Text(), nullable=False),
        sa.Column("checkpoint_revision", sa.Text(), nullable=False),
        sa.Column("adapter_repository_id", sa.Text(), nullable=True),
        sa.Column("adapter_revision", sa.Text(), nullable=True),
        sa.Column("parameter_count", sa.BigInteger(), nullable=True),
        sa.Column("embedding_dim", sa.Integer(), nullable=True),
        sa.Column("distance_metric", sa.String(length=20), nullable=True),
        sa.Column(
            "config",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "btrim(checkpoint_revision) <> '' AND lower(checkpoint_revision) NOT IN ('main','master','latest')",
            name=op.f("ck_model_versions_checkpoint_revision"),
        ),
        sa.CheckConstraint(
            "btrim(model_key) <> '' AND btrim(repository_id) <> ''",
            name=op.f("ck_model_versions_identity_nonblank"),
        ),
        sa.CheckConstraint(
            "role IN ('paper_embedding','chunk_embedding','reranking','extraction','summarization')",
            name=op.f("ck_model_versions_role"),
        ),
        sa.CheckConstraint("embedding_dim > 0", name=op.f("ck_model_versions_embedding_dim")),
        sa.CheckConstraint("parameter_count > 0", name=op.f("ck_model_versions_parameter_count")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model_versions")),
        sa.UniqueConstraint("model_key", name=op.f("uq_model_versions_model_key")),
    )
    op.create_table(
        "papers",
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("abstract", sa.Text(), nullable=True),
        sa.Column("publication_year", sa.SmallInteger(), nullable=True),
        sa.Column("venue_name", sa.Text(), nullable=True),
        sa.Column("venue_type", sa.String(length=30), nullable=True),
        sa.Column("doi", sa.Text(), nullable=True),
        sa.Column("arxiv_id", sa.Text(), nullable=True),
        sa.Column(
            "keywords",
            postgresql.ARRAY(sa.Text()),
            server_default=sa.text("'{}'::text[]"),
            nullable=False,
        ),
        sa.Column("language", sa.String(length=12), nullable=True),
        sa.Column("citation_key", sa.Text(), nullable=True),
        sa.Column("paper_role", sa.String(length=12), server_default="corpus", nullable=False),
        sa.Column("metadata_source", sa.String(length=40), nullable=True),
        sa.Column(
            "metadata_extra",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("btrim(title) <> ''", name=op.f("ck_papers_title_nonblank")),
        sa.CheckConstraint("paper_role IN ('corpus', 'query')", name=op.f("ck_papers_paper_role")),
        sa.CheckConstraint(
            "venue_type IN ('journal','conference','preprint','workshop','other')",
            name=op.f("ck_papers_venue_type"),
        ),
        sa.CheckConstraint(
            "publication_year BETWEEN 1800 AND 2100", name=op.f("ck_papers_publication_year")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_papers")),
        sa.UniqueConstraint("citation_key", name=op.f("uq_papers_citation_key")),
    )
    op.create_index(
        op.f("ix_papers_publication_year"), "papers", ["publication_year"], unique=False
    )
    op.create_index(
        "uq_papers_arxiv_id_normalized",
        "papers",
        [sa.literal_column("lower(arxiv_id)")],
        unique=True,
    )
    op.create_index(
        "uq_papers_doi_normalized", "papers", [sa.literal_column("lower(doi)")], unique=True
    )
    op.create_table(
        "paper_authors",
        sa.Column("paper_id", sa.Uuid(), nullable=False),
        sa.Column("author_position", sa.Integer(), nullable=False),
        sa.Column("full_name", sa.Text(), nullable=False),
        sa.Column("orcid", sa.String(length=25), nullable=True),
        sa.Column("affiliation", sa.Text(), nullable=True),
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("btrim(full_name) <> ''", name=op.f("ck_paper_authors_full_name")),
        sa.CheckConstraint("author_position >= 1", name=op.f("ck_paper_authors_position")),
        sa.ForeignKeyConstraint(
            ["paper_id"],
            ["papers.id"],
            name=op.f("fk_paper_authors_paper_id_papers"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_paper_authors")),
        sa.UniqueConstraint("paper_id", "author_position", name=op.f("uq_paper_authors_paper_id")),
    )
    op.create_index(op.f("ix_paper_authors_paper_id"), "paper_authors", ["paper_id"], unique=False)
    op.create_table(
        "paper_embeddings",
        sa.Column("paper_id", sa.Uuid(), nullable=False),
        sa.Column("model_version_id", sa.Uuid(), nullable=False),
        sa.Column("embedding", VECTOR(), nullable=False),
        sa.Column("source_text_sha256", sa.CHAR(length=64), nullable=False),
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "source_text_sha256 ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_paper_embeddings_source_text_sha256"),
        ),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_versions.id"],
            name=op.f("fk_paper_embeddings_model_version_id_model_versions"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["paper_id"],
            ["papers.id"],
            name=op.f("fk_paper_embeddings_paper_id_papers"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_paper_embeddings")),
        sa.UniqueConstraint(
            "paper_id", "model_version_id", name=op.f("uq_paper_embeddings_paper_id")
        ),
    )
    op.create_index(
        op.f("ix_paper_embeddings_model_version_id"),
        "paper_embeddings",
        ["model_version_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_paper_embeddings_paper_id"), "paper_embeddings", ["paper_id"], unique=False
    )
    op.create_table(
        "paper_files",
        sa.Column("paper_id", sa.Uuid(), nullable=False),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("sha256", sa.CHAR(length=64), nullable=False),
        sa.Column(
            "mime_type", sa.String(length=80), server_default="application/pdf", nullable=False
        ),
        sa.Column("byte_size", sa.BigInteger(), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("license_note", sa.Text(), nullable=True),
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "btrim(relative_path) <> '' AND relative_path !~ '^/' AND relative_path !~ '(^|/)\\.\\.(/|$)'",
            name=op.f("ck_paper_files_relative_path"),
        ),
        sa.CheckConstraint("sha256 ~ '^[0-9a-f]{64}$'", name=op.f("ck_paper_files_sha256")),
        sa.CheckConstraint("byte_size > 0", name=op.f("ck_paper_files_byte_size")),
        sa.CheckConstraint("page_count > 0", name=op.f("ck_paper_files_page_count")),
        sa.ForeignKeyConstraint(
            ["paper_id"],
            ["papers.id"],
            name=op.f("fk_paper_files_paper_id_papers"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_paper_files")),
        sa.UniqueConstraint("sha256", name=op.f("uq_paper_files_sha256")),
    )
    op.create_index(op.f("ix_paper_files_paper_id"), "paper_files", ["paper_id"], unique=False)
    op.create_table(
        "paper_sections",
        sa.Column("paper_id", sa.Uuid(), nullable=False),
        sa.Column("section_order", sa.Integer(), nullable=False),
        sa.Column("heading", sa.Text(), nullable=True),
        sa.Column("section_type", sa.String(length=30), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "section_type IN ('abstract','introduction','related_work','methods','experiments','results','discussion','conclusion','references','other')",
            name=op.f("ck_paper_sections_section_type"),
        ),
        sa.CheckConstraint("page_end >= page_start", name=op.f("ck_paper_sections_page_range")),
        sa.CheckConstraint(
            "page_end IS NULL OR page_end >= 1", name=op.f("ck_paper_sections_page_end")
        ),
        sa.CheckConstraint(
            "page_start IS NULL OR page_start >= 1", name=op.f("ck_paper_sections_page_start")
        ),
        sa.CheckConstraint("section_order >= 0", name=op.f("ck_paper_sections_section_order")),
        sa.ForeignKeyConstraint(
            ["paper_id"],
            ["papers.id"],
            name=op.f("fk_paper_sections_paper_id_papers"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_paper_sections")),
        sa.UniqueConstraint("id", "paper_id", name=op.f("uq_paper_sections_id")),
        sa.UniqueConstraint("paper_id", "section_order", name=op.f("uq_paper_sections_paper_id")),
    )
    op.create_index(
        op.f("ix_paper_sections_paper_id"), "paper_sections", ["paper_id"], unique=False
    )
    op.create_table(
        "chunks",
        sa.Column("paper_id", sa.Uuid(), nullable=False),
        sa.Column("section_id", sa.Uuid(), nullable=True),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("chunk_type", sa.String(length=24), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_sha256", sa.CHAR(length=64), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column(
            "source_spans",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("chunker_version", sa.Text(), nullable=True),
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("btrim(content) <> ''", name=op.f("ck_chunks_content")),
        sa.CheckConstraint(
            "chunk_type IN ('abstract','body','references')", name=op.f("ck_chunks_chunk_type")
        ),
        sa.CheckConstraint(
            "content_sha256 ~ '^[0-9a-f]{64}$'", name=op.f("ck_chunks_content_sha256")
        ),
        sa.CheckConstraint(
            "jsonb_typeof(source_spans) = 'array'", name=op.f("ck_chunks_source_spans")
        ),
        sa.CheckConstraint("chunk_index >= 0", name=op.f("ck_chunks_chunk_index")),
        sa.CheckConstraint("page_end >= page_start", name=op.f("ck_chunks_page_range")),
        sa.CheckConstraint("page_end IS NULL OR page_end >= 1", name=op.f("ck_chunks_page_end")),
        sa.CheckConstraint(
            "page_start IS NULL OR page_start >= 1", name=op.f("ck_chunks_page_start")
        ),
        sa.CheckConstraint("token_count >= 0", name=op.f("ck_chunks_token_count")),
        sa.ForeignKeyConstraint(
            ["paper_id"], ["papers.id"], name=op.f("fk_chunks_paper_id_papers"), ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["section_id", "paper_id"],
            ["paper_sections.id", "paper_sections.paper_id"],
            name=op.f("fk_chunks_section_id_paper_sections"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chunks")),
        sa.UniqueConstraint("paper_id", "chunk_index", name=op.f("uq_chunks_paper_id")),
    )
    op.create_index(op.f("ix_chunks_paper_id"), "chunks", ["paper_id"], unique=False)
    op.create_index(op.f("ix_chunks_section_id"), "chunks", ["section_id"], unique=False)
    op.create_table(
        "ingestion_runs",
        sa.Column("paper_file_id", sa.Uuid(), nullable=True),
        sa.Column("status", sa.String(length=16), server_default="pending", nullable=False),
        sa.Column("stage", sa.String(length=40), nullable=True),
        sa.Column("parser_name", sa.Text(), nullable=True),
        sa.Column("parser_version", sa.Text(), nullable=True),
        sa.Column(
            "config",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "warnings",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.CheckConstraint(
            "status IN ('pending','running','succeeded','failed')",
            name=op.f("ck_ingestion_runs_status"),
        ),
        sa.CheckConstraint("finished_at >= started_at", name=op.f("ck_ingestion_runs_time_range")),
        sa.ForeignKeyConstraint(
            ["paper_file_id"],
            ["paper_files.id"],
            name=op.f("fk_ingestion_runs_paper_file_id_paper_files"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ingestion_runs")),
    )
    op.create_index(
        op.f("ix_ingestion_runs_paper_file_id"), "ingestion_runs", ["paper_file_id"], unique=False
    )
    op.create_index(op.f("ix_ingestion_runs_status"), "ingestion_runs", ["status"], unique=False)
    op.create_table(
        "chunk_embeddings",
        sa.Column("chunk_id", sa.Uuid(), nullable=False),
        sa.Column("model_version_id", sa.Uuid(), nullable=False),
        sa.Column("embedding", VECTOR(), nullable=False),
        sa.Column("source_text_sha256", sa.CHAR(length=64), nullable=False),
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "source_text_sha256 ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_chunk_embeddings_source_text_sha256"),
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["chunks.id"],
            name=op.f("fk_chunk_embeddings_chunk_id_chunks"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_versions.id"],
            name=op.f("fk_chunk_embeddings_model_version_id_model_versions"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chunk_embeddings")),
        sa.UniqueConstraint(
            "chunk_id", "model_version_id", name=op.f("uq_chunk_embeddings_chunk_id")
        ),
    )
    op.create_index(
        op.f("ix_chunk_embeddings_chunk_id"), "chunk_embeddings", ["chunk_id"], unique=False
    )
    op.create_index(
        op.f("ix_chunk_embeddings_model_version_id"),
        "chunk_embeddings",
        ["model_version_id"],
        unique=False,
    )
    op.execute("""
        CREATE FUNCTION rw_touch_updated_at() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN NEW.updated_at = clock_timestamp(); RETURN NEW; END $$
    """)
    op.execute(
        "CREATE TRIGGER papers_updated_at BEFORE UPDATE ON papers "
        "FOR EACH ROW EXECUTE FUNCTION rw_touch_updated_at()"
    )
    op.execute("""
        CREATE FUNCTION rw_validate_embedding() RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE expected_dim integer; expected_role text;
        BEGIN
            SELECT embedding_dim, role INTO expected_dim, expected_role
            FROM model_versions WHERE id = NEW.model_version_id FOR SHARE;
            IF NOT FOUND THEN
                RAISE foreign_key_violation USING MESSAGE = 'Unknown model version';
            END IF;
            IF expected_dim IS NULL OR vector_dims(NEW.embedding) <> expected_dim
               OR expected_role <> TG_ARGV[0] THEN
                RAISE check_violation USING MESSAGE = 'Embedding dimension or model role mismatch';
            END IF;
            RETURN NEW;
        END $$
    """)
    for table, role in (
        ("paper_embeddings", "paper_embedding"),
        ("chunk_embeddings", "chunk_embedding"),
    ):
        op.execute(
            f"CREATE TRIGGER {table}_validate BEFORE INSERT OR UPDATE ON {table} "
            f"FOR EACH ROW EXECUTE FUNCTION rw_validate_embedding('{role}')"
        )
    op.execute("""
        CREATE FUNCTION rw_model_version_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF NEW IS DISTINCT FROM OLD THEN
                RAISE check_violation USING MESSAGE = 'Model versions are immutable; register a new version';
            END IF;
            RETURN NEW;
        END $$
    """)
    op.execute(
        "CREATE TRIGGER model_versions_immutable BEFORE UPDATE ON model_versions "
        "FOR EACH ROW EXECUTE FUNCTION rw_model_version_immutable()"
    )


def downgrade() -> None:
    """Downgrade schema."""
    # The vector extension may be shared; intentionally retain it.
    op.drop_index(op.f("ix_chunk_embeddings_model_version_id"), table_name="chunk_embeddings")
    op.drop_index(op.f("ix_chunk_embeddings_chunk_id"), table_name="chunk_embeddings")
    op.drop_table("chunk_embeddings")
    op.drop_index(op.f("ix_ingestion_runs_status"), table_name="ingestion_runs")
    op.drop_index(op.f("ix_ingestion_runs_paper_file_id"), table_name="ingestion_runs")
    op.drop_table("ingestion_runs")
    op.drop_index(op.f("ix_chunks_section_id"), table_name="chunks")
    op.drop_index(op.f("ix_chunks_paper_id"), table_name="chunks")
    op.drop_table("chunks")
    op.drop_index(op.f("ix_paper_sections_paper_id"), table_name="paper_sections")
    op.drop_table("paper_sections")
    op.drop_index(op.f("ix_paper_files_paper_id"), table_name="paper_files")
    op.drop_table("paper_files")
    op.drop_index(op.f("ix_paper_embeddings_paper_id"), table_name="paper_embeddings")
    op.drop_index(op.f("ix_paper_embeddings_model_version_id"), table_name="paper_embeddings")
    op.drop_table("paper_embeddings")
    op.drop_index(op.f("ix_paper_authors_paper_id"), table_name="paper_authors")
    op.drop_table("paper_authors")
    op.drop_index("uq_papers_doi_normalized", table_name="papers")
    op.drop_index("uq_papers_arxiv_id_normalized", table_name="papers")
    op.drop_index(op.f("ix_papers_publication_year"), table_name="papers")
    op.drop_table("papers")
    op.drop_table("model_versions")
    op.execute("DROP FUNCTION rw_touch_updated_at()")
    op.execute("DROP FUNCTION rw_validate_embedding()")
    op.execute("DROP FUNCTION rw_model_version_immutable()")
