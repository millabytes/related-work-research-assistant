"""Phase-0 persistence models. Application tables are created only by Alembic."""

import re
from datetime import datetime
from typing import Any
from uuid import UUID

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    CHAR,
    BigInteger,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column, validates

from related_work_assistant.database.base import Base


class Identity:
    id: Mapped[UUID] = mapped_column(primary_key=True, server_default=text("gen_random_uuid()"))


class Created:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


def page_checks() -> tuple[CheckConstraint, ...]:
    return (
        CheckConstraint("page_start IS NULL OR page_start >= 1", name="page_start"),
        CheckConstraint("page_end IS NULL OR page_end >= 1", name="page_end"),
        CheckConstraint("page_end >= page_start", name="page_range"),
    )


class Paper(Identity, Created, Base):
    __tablename__ = "papers"
    title: Mapped[str | None] = mapped_column(Text)
    abstract: Mapped[str | None] = mapped_column(Text)
    publication_year: Mapped[int | None] = mapped_column(SmallInteger, index=True)
    venue_name: Mapped[str | None] = mapped_column(Text)
    venue_type: Mapped[str | None] = mapped_column(String(30))
    doi: Mapped[str | None] = mapped_column(Text)
    arxiv_id: Mapped[str | None] = mapped_column(Text)
    keywords: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default=text("'{}'::text[]"))
    language: Mapped[str | None] = mapped_column(String(12))
    citation_key: Mapped[str | None] = mapped_column(Text, unique=True)
    paper_role: Mapped[str] = mapped_column(String(12), server_default="corpus")
    metadata_source: Mapped[str | None] = mapped_column(String(40))
    metadata_extra: Mapped[dict[str, Any]] = mapped_column(
        JSONB, server_default=text("'{}'::jsonb")
    )
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (
        CheckConstraint("btrim(title) <> ''", name="title_nonblank"),
        CheckConstraint("publication_year BETWEEN 1800 AND 2100", name="publication_year"),
        CheckConstraint("paper_role IN ('corpus', 'query')", name="paper_role"),
        CheckConstraint(
            "venue_type IN ('journal','conference','preprint','workshop','other')",
            name="venue_type",
        ),
        Index("uq_papers_doi_normalized", func.lower(doi), unique=True),
        Index("uq_papers_arxiv_id_normalized", func.lower(arxiv_id), unique=True),
    )

    @validates("doi", "arxiv_id")
    def normalize_identifier(self, key: str, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().lower()
        if key == "doi":
            value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value)
        else:
            value = re.sub(r"^(?:https?://arxiv\.org/abs/|arxiv:\s*)", "", value)
        return value or None


class PaperAuthor(Identity, Created, Base):
    __tablename__ = "paper_authors"
    paper_id: Mapped[UUID] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    author_position: Mapped[int] = mapped_column(Integer)
    first_name: Mapped[str | None] = mapped_column(Text)
    last_name: Mapped[str | None] = mapped_column(Text)
    full_name: Mapped[str] = mapped_column(Text)
    orcid: Mapped[str | None] = mapped_column(String(25))
    affiliation: Mapped[str | None] = mapped_column(Text)
    __table_args__ = (
        UniqueConstraint("paper_id", "author_position"),
        CheckConstraint("author_position >= 1", name="position"),
        CheckConstraint("btrim(full_name) <> ''", name="full_name"),
    )


class PaperFile(Identity, Created, Base):
    __tablename__ = "paper_files"
    paper_id: Mapped[UUID] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    relative_path: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(CHAR(64), unique=True)
    mime_type: Mapped[str] = mapped_column(String(80), server_default="application/pdf")
    byte_size: Mapped[int] = mapped_column(BigInteger)
    page_count: Mapped[int | None] = mapped_column(Integer)
    source_url: Mapped[str | None] = mapped_column(Text)
    license_note: Mapped[str | None] = mapped_column(Text)
    __table_args__ = (
        CheckConstraint("sha256 ~ '^[0-9a-f]{64}$'", name="sha256"),
        CheckConstraint("byte_size > 0", name="byte_size"),
        CheckConstraint("page_count > 0", name="page_count"),
        CheckConstraint(
            "btrim(relative_path) <> '' AND relative_path !~ '^/' AND relative_path !~ '(^|/)\\.\\.(/|$)'",
            name="relative_path",
        ),
    )


class PaperSection(Identity, Created, Base):
    __tablename__ = "paper_sections"
    paper_id: Mapped[UUID] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    source_file_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("paper_files.id", ondelete="CASCADE"), index=True
    )
    source_sha256: Mapped[str | None] = mapped_column(CHAR(64))
    parser_version: Mapped[str | None] = mapped_column(Text)
    section_order: Mapped[int] = mapped_column(Integer)
    heading: Mapped[str | None] = mapped_column(Text)
    section_type: Mapped[str] = mapped_column(String(30))
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    __table_args__ = (
        UniqueConstraint("paper_id", "section_order"),
        UniqueConstraint("id", "paper_id"),
        CheckConstraint("section_order >= 0", name="section_order"),
        CheckConstraint(
            "source_sha256 IS NULL OR source_sha256 ~ '^[0-9a-f]{64}$'",
            name="source_sha256",
        ),
        CheckConstraint(
            "section_type IN ('abstract','introduction','related_work','methods','experiments','results','discussion','conclusion','references','other')",
            name="section_type",
        ),
        *page_checks(),
    )


class Chunk(Identity, Created, Base):
    __tablename__ = "chunks"
    paper_id: Mapped[UUID] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    section_id: Mapped[UUID | None] = mapped_column(index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    chunk_type: Mapped[str] = mapped_column(String(24))
    content: Mapped[str] = mapped_column(Text)
    content_sha256: Mapped[str] = mapped_column(CHAR(64))
    token_count: Mapped[int | None] = mapped_column(Integer)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    source_spans: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, server_default=text("'[]'::jsonb")
    )
    chunker_version: Mapped[str | None] = mapped_column(Text)
    __table_args__ = (
        ForeignKeyConstraint(
            ["section_id", "paper_id"],
            ["paper_sections.id", "paper_sections.paper_id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("paper_id", "chunk_index"),
        CheckConstraint("chunk_index >= 0", name="chunk_index"),
        CheckConstraint("chunk_type IN ('abstract','body','references')", name="chunk_type"),
        CheckConstraint("btrim(content) <> ''", name="content"),
        CheckConstraint("content_sha256 ~ '^[0-9a-f]{64}$'", name="content_sha256"),
        CheckConstraint("token_count >= 0", name="token_count"),
        CheckConstraint("jsonb_typeof(source_spans) = 'array'", name="source_spans"),
        *page_checks(),
    )


class ModelVersion(Identity, Created, Base):
    __tablename__ = "model_versions"
    model_key: Mapped[str] = mapped_column(Text, unique=True)
    role: Mapped[str] = mapped_column(String(30))
    provider: Mapped[str | None] = mapped_column(Text)
    repository_id: Mapped[str] = mapped_column(Text)
    checkpoint_revision: Mapped[str] = mapped_column(Text)
    adapter_repository_id: Mapped[str | None] = mapped_column(Text)
    adapter_revision: Mapped[str | None] = mapped_column(Text)
    parameter_count: Mapped[int | None] = mapped_column(BigInteger)
    embedding_dim: Mapped[int | None] = mapped_column(Integer)
    distance_metric: Mapped[str | None] = mapped_column(String(20))
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    __table_args__ = (
        CheckConstraint(
            "role IN ('paper_embedding','chunk_embedding','reranking','extraction','summarization','generation')",
            name="role",
        ),
        CheckConstraint("parameter_count > 0", name="parameter_count"),
        CheckConstraint("embedding_dim > 0", name="embedding_dim"),
        CheckConstraint(
            "btrim(model_key) <> '' AND btrim(repository_id) <> ''", name="identity_nonblank"
        ),
        CheckConstraint(
            "btrim(checkpoint_revision) <> '' AND lower(checkpoint_revision) NOT IN ('main','master','latest')",
            name="checkpoint_revision",
        ),
    )


class PaperEmbedding(Identity, Created, Base):
    __tablename__ = "paper_embeddings"
    paper_id: Mapped[UUID] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    model_version_id: Mapped[UUID] = mapped_column(
        ForeignKey("model_versions.id", ondelete="RESTRICT"), index=True
    )
    embedding: Mapped[list[float]] = mapped_column(VECTOR())
    source_text_sha256: Mapped[str] = mapped_column(CHAR(64))
    __table_args__ = (
        UniqueConstraint("paper_id", "model_version_id"),
        CheckConstraint("source_text_sha256 ~ '^[0-9a-f]{64}$'", name="source_text_sha256"),
    )


class ChunkEmbedding(Identity, Created, Base):
    __tablename__ = "chunk_embeddings"
    chunk_id: Mapped[UUID] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"), index=True)
    model_version_id: Mapped[UUID] = mapped_column(
        ForeignKey("model_versions.id", ondelete="RESTRICT"), index=True
    )
    embedding: Mapped[list[float]] = mapped_column(VECTOR())
    source_text_sha256: Mapped[str] = mapped_column(CHAR(64))
    __table_args__ = (
        UniqueConstraint("chunk_id", "model_version_id"),
        CheckConstraint("source_text_sha256 ~ '^[0-9a-f]{64}$'", name="source_text_sha256"),
    )


class IngestionRun(Identity, Base):
    __tablename__ = "ingestion_runs"
    paper_file_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("paper_files.id", ondelete="SET NULL"), index=True
    )
    status: Mapped[str] = mapped_column(String(16), index=True, server_default="pending")
    stage: Mapped[str | None] = mapped_column(String(40))
    parser_name: Mapped[str | None] = mapped_column(Text)
    parser_version: Mapped[str | None] = mapped_column(Text)
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    warnings: Mapped[list[Any]] = mapped_column(JSONB, server_default=text("'[]'::jsonb"))
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    __table_args__ = (
        CheckConstraint("status IN ('pending','running','succeeded','failed')", name="status"),
        CheckConstraint("finished_at >= started_at", name="time_range"),
    )


class RelatedWorkGeneration(Identity, Created, Base):
    __tablename__ = "related_work_generations"
    paper_id: Mapped[UUID] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    model_key: Mapped[str] = mapped_column(String(24), index=True)
    model_version_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("model_versions.id", ondelete="RESTRICT"), index=True
    )
    model_repository_id: Mapped[str] = mapped_column(Text)
    model_revision: Mapped[str] = mapped_column(Text)
    prompt_version: Mapped[str] = mapped_column(Text)
    generation_version: Mapped[str] = mapped_column(Text)
    generation_config: Mapped[dict[str, Any]] = mapped_column(JSONB)
    cache_key: Mapped[str] = mapped_column(CHAR(64), index=True)
    evidence_hash: Mapped[str] = mapped_column(CHAR(64))
    input_hash: Mapped[str] = mapped_column(CHAR(64))
    evidence_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB)
    generated_text: Mapped[str | None] = mapped_column(Text)
    citations: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default=text("'{}'::text[]"))
    source_references: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, server_default=text("'[]'::jsonb")
    )
    input_token_count: Mapped[int | None] = mapped_column(Integer)
    output_token_count: Mapped[int | None] = mapped_column(Integer)
    context_limit: Mapped[int | None] = mapped_column(Integer)
    inference_latency_ms: Mapped[float | None] = mapped_column(Float)
    peak_memory_mb: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(24), index=True)
    error_message: Mapped[str | None] = mapped_column(Text)
    finish_reason: Mapped[str | None] = mapped_column(String(24))
    validation_errors: Mapped[list[str]] = mapped_column(JSONB, server_default=text("'[]'::jsonb"))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    __table_args__ = (
        CheckConstraint(
            "model_key IN ('primera','gemma','qwen','qwen_lora')", name="model_key"
        ),
        CheckConstraint(
            "status IN ('completed','context_overflow','invalid','failed')", name="status"
        ),
        CheckConstraint("cache_key ~ '^[0-9a-f]{64}$'", name="cache_key"),
        CheckConstraint("evidence_hash ~ '^[0-9a-f]{64}$'", name="evidence_hash"),
        CheckConstraint("input_hash ~ '^[0-9a-f]{64}$'", name="input_hash"),
        CheckConstraint("inference_latency_ms >= 0", name="inference_latency_ms"),
        CheckConstraint("peak_memory_mb >= 0", name="peak_memory_mb"),
    )


class RelatedWorkJob(Identity, Created, Base):
    __tablename__ = "related_work_jobs"
    paper_id: Mapped[UUID] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    model_key: Mapped[str] = mapped_column(String(24), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True, server_default="pending")
    force_regenerate: Mapped[bool] = mapped_column(server_default=text("false"))
    cache_key: Mapped[str] = mapped_column(CHAR(64))
    generation_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("related_work_generations.id", ondelete="SET NULL"), index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)
    __table_args__ = (
        CheckConstraint(
            "model_key IN ('primera','gemma','qwen','qwen_lora')", name="model_key"
        ),
        CheckConstraint("status IN ('pending','running','completed','failed')", name="status"),
        CheckConstraint("cache_key ~ '^[0-9a-f]{64}$'", name="cache_key"),
        Index(
            "uq_related_work_jobs_active_model",
            "paper_id",
            "model_key",
            unique=True,
            postgresql_where=text("status IN ('pending','running')"),
        ),
    )
