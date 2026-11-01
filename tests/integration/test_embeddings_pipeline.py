from dataclasses import replace

import numpy as np
import pytest
from sqlalchemy import func, select

from related_work_assistant.database.models import (
    Chunk,
    ChunkEmbedding,
    ModelVersion,
    Paper,
    PaperEmbedding,
)
from related_work_assistant.database.session import session_scope
from related_work_assistant.embeddings.encoders import ModelDescriptor
from related_work_assistant.pipelines.embeddings import (
    embed_chunks,
    embed_papers,
    embedding_status,
)

pytestmark = pytest.mark.integration
HASH = "c" * 64


@pytest.fixture
def embedding_connection(db_engine):
    with db_engine.connect() as connection:
        transaction = connection.begin()
        try:
            yield connection
        finally:
            transaction.rollback()


class FakeEncoder:
    def __init__(self, descriptor):
        self.descriptor = descriptor
        self.calls = 0

    def encode(self, texts, batch_size):
        self.calls += 1
        rows = []
        for index, _text in enumerate(texts, start=1):
            vector = np.zeros(self.descriptor.embedding_dim, dtype=np.float32)
            vector[0] = index
            rows.append(vector)
        return np.stack(rows)


def paper_descriptor(**changes):
    base = ModelDescriptor(
        role="paper_embedding",
        repository_id="local/specter-fixture",
        checkpoint_revision="a" * 40,
        adapter_repository_id="local/adapter-fixture",
        adapter_revision="b" * 40,
        embedding_dim=768,
        parameter_count=100,
        config={"pooling": "cls"},
    )
    return replace(base, **changes)


def chunk_descriptor(**changes):
    base = ModelDescriptor(
        role="chunk_embedding",
        repository_id="local/bge-fixture",
        checkpoint_revision="d" * 40,
        embedding_dim=384,
        parameter_count=50,
        config={"normalized": True},
    )
    return replace(base, **changes)


def seed(session):
    first = Paper(title="First paper", abstract="First abstract")
    second = Paper(title="Second paper", abstract="Second abstract")
    missing = Paper(title="No abstract")
    session.add_all([first, second, missing])
    session.flush()
    chunks = [
        Chunk(
            paper_id=paper.id,
            chunk_index=0,
            chunk_type="body",
            content=f"Chunk for {paper.title}",
            content_sha256=HASH,
        )
        for paper in (first, second)
    ]
    session.add_all(chunks)
    session.flush()
    return first, second, missing, chunks


def test_paper_embeddings_idempotence_updates_and_versions(embedding_connection):
    with session_scope(embedding_connection) as session:
        first, second, _missing, _chunks = seed(session)
    encoder = FakeEncoder(paper_descriptor())

    initial = embed_papers(embedding_connection, encoder, "[SEP]", batch_size=1)
    assert (initial.selected, initial.embedded, initial.ineligible) == (3, 2, 1)
    assert encoder.calls == 2
    assert embed_papers(embedding_connection, encoder, "[SEP]").skipped == 2
    assert encoder.calls == 2

    with embedding_connection.begin_nested():
        embedding_connection.execute(
            Paper.__table__.update().where(Paper.id == first.id).values(abstract="Changed")
        )
    changed = embed_papers(embedding_connection, encoder, "[SEP]")
    assert (changed.embedded, changed.updated, changed.skipped) == (1, 1, 1)

    second_encoder = FakeEncoder(
        paper_descriptor(checkpoint_revision="e" * 40)
    )
    embed_papers(embedding_connection, second_encoder, "[SEP]", paper_ids=[second.id])
    assert embedding_connection.scalar(select(func.count()).select_from(ModelVersion)) == 2
    assert embedding_connection.scalar(select(func.count()).select_from(PaperEmbedding)) == 3
    dimensions = embedding_connection.execute(
        select(func.vector_dims(PaperEmbedding.embedding))
    ).scalars().all()
    assert dimensions == [768, 768, 768]


def test_chunk_embeddings_batching_dimensions_and_filter(embedding_connection):
    with session_scope(embedding_connection) as session:
        first, _second, _missing, chunks = seed(session)
    encoder = FakeEncoder(chunk_descriptor())
    initial = embed_chunks(embedding_connection, encoder, paper_ids=[first.id], batch_size=8)
    assert (initial.selected, initial.embedded, initial.skipped) == (1, 1, 0)
    assert embed_chunks(embedding_connection, encoder, paper_ids=[first.id]).skipped == 1

    assert embedding_connection.scalar(select(func.count()).select_from(ChunkEmbedding)) == 1
    assert embedding_connection.scalar(
        select(func.vector_dims(ChunkEmbedding.embedding)).where(
            ChunkEmbedding.chunk_id == chunks[0].id
        )
    ) == 384
    status = embedding_status(embedding_connection)
    assert status[0]["role"] == "chunk_embedding"
    assert status[0]["embedding_dim"] == 384
    assert status[0]["row_count"] == 1
