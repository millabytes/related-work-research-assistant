"""Idempotent paper and chunk embedding persistence."""

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Protocol
from uuid import UUID

import numpy as np
from sqlalchemy import Engine, func, select
from sqlalchemy.dialects.postgresql import insert

from related_work_assistant.database.models import (
    Chunk,
    ChunkEmbedding,
    ModelVersion,
    Paper,
    PaperEmbedding,
)
from related_work_assistant.database.session import session_scope
from related_work_assistant.embeddings.encoders import ModelDescriptor


class Encoder(Protocol):
    descriptor: ModelDescriptor

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray: ...


@dataclass
class EmbeddingResult:
    model_key: str
    model_version_id: str
    selected: int
    embedded: int
    updated: int
    skipped: int
    ineligible: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def text_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def paper_input(title: str, abstract: str, separator: str) -> str:
    return f"{title.strip()}{separator}{abstract.strip()}"


def model_key(descriptor: ModelDescriptor) -> str:
    identity = {
        "role": descriptor.role,
        "repository_id": descriptor.repository_id,
        "checkpoint_revision": descriptor.checkpoint_revision,
        "adapter_repository_id": descriptor.adapter_repository_id,
        "adapter_revision": descriptor.adapter_revision,
        "embedding_dim": descriptor.embedding_dim,
        "config": descriptor.config,
    }
    digest = sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return f"{descriptor.role}:{descriptor.repository_id}:{digest[:16]}"


def register_model(engine: Engine, descriptor: ModelDescriptor) -> tuple[UUID, str]:
    key = model_key(descriptor)
    with session_scope(engine) as session:
        row = session.scalar(select(ModelVersion).where(ModelVersion.model_key == key))
        if row is None:
            row = ModelVersion(
                model_key=key,
                role=descriptor.role,
                provider=descriptor.provider,
                repository_id=descriptor.repository_id,
                checkpoint_revision=descriptor.checkpoint_revision,
                adapter_repository_id=descriptor.adapter_repository_id,
                adapter_revision=descriptor.adapter_revision,
                parameter_count=descriptor.parameter_count,
                embedding_dim=descriptor.embedding_dim,
                distance_metric=descriptor.distance_metric,
                config=descriptor.config,
            )
            session.add(row)
            session.flush()
        return row.id, key


def _batches(items: list[tuple[UUID, str, str, bool]], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def embed_papers(
    engine: Engine,
    encoder: Encoder,
    separator: str,
    paper_ids: list[UUID] | None = None,
    batch_size: int = 8,
    limit: int | None = None,
) -> EmbeddingResult:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model_id, key = register_model(engine, encoder.descriptor)
    with session_scope(engine) as session:
        statement = (
            select(
                Paper.id,
                Paper.title,
                Paper.abstract,
                PaperEmbedding.source_text_sha256,
            )
            .outerjoin(
                PaperEmbedding,
                (PaperEmbedding.paper_id == Paper.id)
                & (PaperEmbedding.model_version_id == model_id),
            )
            .order_by(Paper.id)
        )
        if paper_ids:
            statement = statement.where(Paper.id.in_(paper_ids))
        rows = list(session.execute(statement))
    if limit is not None:
        rows = rows[:limit]

    pending: list[tuple[UUID, str, str, bool]] = []
    skipped = ineligible = 0
    for paper_id, title, abstract, stored_hash in rows:
        if not title or not title.strip() or not abstract or not abstract.strip():
            ineligible += 1
            continue
        content = paper_input(title, abstract, separator)
        digest = text_hash(content)
        if stored_hash == digest:
            skipped += 1
        else:
            pending.append((paper_id, content, digest, stored_hash is not None))

    embedded = updated = 0
    for batch in _batches(pending, batch_size):
        vectors = encoder.encode([item[1] for item in batch], batch_size=batch_size)
        with session_scope(engine) as session:
            for (paper_id, _content, digest, existed), vector in zip(batch, vectors, strict=True):
                statement = insert(PaperEmbedding).values(
                    paper_id=paper_id,
                    model_version_id=model_id,
                    embedding=vector.tolist(),
                    source_text_sha256=digest,
                )
                session.execute(
                    statement.on_conflict_do_update(
                        index_elements=["paper_id", "model_version_id"],
                        set_={"embedding": statement.excluded.embedding,
                              "source_text_sha256": statement.excluded.source_text_sha256},
                    )
                )
                embedded += 1
                updated += int(existed)
    return EmbeddingResult(key, str(model_id), len(rows), embedded, updated, skipped, ineligible)


def embed_chunks(
    engine: Engine,
    encoder: Encoder,
    paper_ids: list[UUID] | None = None,
    batch_size: int = 32,
    limit: int | None = None,
) -> EmbeddingResult:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model_id, key = register_model(engine, encoder.descriptor)
    with session_scope(engine) as session:
        statement = (
            select(Chunk.id, Chunk.content, ChunkEmbedding.source_text_sha256)
            .outerjoin(
                ChunkEmbedding,
                (ChunkEmbedding.chunk_id == Chunk.id)
                & (ChunkEmbedding.model_version_id == model_id),
            )
            .order_by(Chunk.paper_id, Chunk.chunk_index)
        )
        if paper_ids:
            statement = statement.where(Chunk.paper_id.in_(paper_ids))
        rows = list(session.execute(statement))
    if limit is not None:
        rows = rows[:limit]

    pending: list[tuple[UUID, str, str, bool]] = []
    skipped = 0
    for chunk_id, content, stored_hash in rows:
        digest = text_hash(content)
        if stored_hash == digest:
            skipped += 1
        else:
            pending.append((chunk_id, content, digest, stored_hash is not None))

    embedded = updated = 0
    for batch in _batches(pending, batch_size):
        vectors = encoder.encode([item[1] for item in batch], batch_size=batch_size)
        with session_scope(engine) as session:
            for (chunk_id, _content, digest, existed), vector in zip(batch, vectors, strict=True):
                statement = insert(ChunkEmbedding).values(
                    chunk_id=chunk_id,
                    model_version_id=model_id,
                    embedding=vector.tolist(),
                    source_text_sha256=digest,
                )
                session.execute(
                    statement.on_conflict_do_update(
                        index_elements=["chunk_id", "model_version_id"],
                        set_={"embedding": statement.excluded.embedding,
                              "source_text_sha256": statement.excluded.source_text_sha256},
                    )
                )
                embedded += 1
                updated += int(existed)
    return EmbeddingResult(key, str(model_id), len(rows), embedded, updated, skipped, 0)


def embedding_status(engine: Engine) -> list[dict[str, object]]:
    with session_scope(engine) as session:
        paper_counts = dict(
            session.execute(
                select(PaperEmbedding.model_version_id, func.count()).group_by(
                    PaperEmbedding.model_version_id
                )
            ).all()
        )
        chunk_counts = dict(
            session.execute(
                select(ChunkEmbedding.model_version_id, func.count()).group_by(
                    ChunkEmbedding.model_version_id
                )
            ).all()
        )
        models = list(session.scalars(select(ModelVersion).order_by(ModelVersion.created_at)))
        return [
            {
                "model_key": model.model_key,
                "role": model.role,
                "repository_id": model.repository_id,
                "checkpoint_revision": model.checkpoint_revision,
                "adapter_revision": model.adapter_revision,
                "embedding_dim": model.embedding_dim,
                "row_count": paper_counts.get(model.id, chunk_counts.get(model.id, 0)),
            }
            for model in models
            if model.role in {"paper_embedding", "chunk_embedding"}
        ]
