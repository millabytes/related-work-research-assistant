from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select, text, update
from sqlalchemy.exc import IntegrityError

from related_work_assistant.api.main import app
from related_work_assistant.database.models import (
    Chunk,
    ChunkEmbedding,
    IngestionRun,
    ModelVersion,
    Paper,
    PaperAuthor,
    PaperEmbedding,
    PaperFile,
    PaperSection,
)
from related_work_assistant.database.session import session_scope

pytestmark = pytest.mark.integration
HASH = "a" * 64


def paper_tree(s):
    paper = Paper(title="Synthetic paper", doi=" HTTPS://DOI.ORG/10.1234/TEST ")
    s.add(paper)
    s.flush()
    authors = [
        PaperAuthor(paper_id=paper.id, author_position=n, full_name=f"Author {n}") for n in (1, 2)
    ]
    file = PaperFile(paper_id=paper.id, relative_path="synthetic.pdf", sha256=HASH, byte_size=100)
    section = PaperSection(
        paper_id=paper.id, section_order=0, section_type="methods", page_start=1, page_end=2
    )
    s.add_all([*authors, file, section])
    s.flush()
    chunk = Chunk(
        paper_id=paper.id,
        section_id=section.id,
        chunk_index=0,
        chunk_type="body",
        content="Synthetic source text.",
        content_sha256=HASH,
    )
    s.add(chunk)
    s.flush()
    return paper, file, section, chunk


def model(s, key="synthetic", dim=3, role="chunk_embedding"):
    row = ModelVersion(
        model_key=key,
        role=role,
        repository_id="local/synthetic",
        checkpoint_revision="sha256:synthetic-fixture",
        embedding_dim=dim,
    )
    s.add(row)
    s.flush()
    return row


def rejects(s, row):
    with pytest.raises(IntegrityError), s.begin_nested():
        s.add(row)
        s.flush()


def test_constraints_and_provenance(session):
    s = session
    p, f, sec, c = paper_tree(s)
    assert p.doi == "10.1234/test"
    assert list(
        s.scalars(select(PaperAuthor.author_position).order_by(PaperAuthor.author_position))
    ) == [1, 2]
    rejects(s, PaperAuthor(paper_id=p.id, author_position=1, full_name="Duplicate"))
    rejects(s, PaperFile(paper_id=p.id, relative_path="other.pdf", sha256=HASH, byte_size=1))
    rejects(s, Paper(title="duplicate DOI", doi="10.1234/TEST"))
    rejects(s, PaperAuthor(paper_id=uuid4(), author_position=1, full_name="Orphan"))
    other = Paper(title="Other")
    s.add(other)
    s.flush()
    rejects(
        s,
        Chunk(
            paper_id=other.id,
            section_id=sec.id,
            chunk_index=0,
            chunk_type="body",
            content="wrong paper",
            content_sha256=HASH,
        ),
    )
    old = p.updated_at
    s.execute(update(Paper).where(Paper.id == p.id).values(title="Updated"))
    s.refresh(p)
    assert p.updated_at > old


@pytest.mark.parametrize(
    "values",
    [
        {"title": " "},
        {"title": "Valid", "publication_year": 1700},
        {"title": "Valid", "paper_role": "bad"},
        {"title": "Valid", "venue_type": "bad"},
    ],
)
def test_paper_checks(session, values):
    rejects(session, Paper(**values))


def test_vectors_and_cascades(session):
    s = session
    p, f, sec, c = paper_tree(s)
    m = model(s)
    m4 = model(s, "four", 4)
    mp = model(s, "paper", 3, "paper_embedding")
    second = Chunk(
        paper_id=p.id,
        chunk_index=1,
        chunk_type="body",
        content="Second source.",
        content_sha256=HASH,
    )
    s.add(second)
    s.flush()
    s.add_all(
        [
            ChunkEmbedding(
                chunk_id=c.id,
                model_version_id=m.id,
                embedding=[0.1, 0.2, 0.3],
                source_text_sha256=HASH,
            ),
            ChunkEmbedding(
                chunk_id=second.id,
                model_version_id=m.id,
                embedding=[1, 0, 0],
                source_text_sha256=HASH,
            ),
            ChunkEmbedding(
                chunk_id=c.id,
                model_version_id=m4.id,
                embedding=[1, 0, 0, 0],
                source_text_sha256=HASH,
            ),
            PaperEmbedding(
                paper_id=p.id, model_version_id=mp.id, embedding=[1, 0, 0], source_text_sha256=HASH
            ),
        ]
    )
    run = IngestionRun(paper_file_id=f.id, status="failed", error_message="Synthetic failure")
    s.add(run)
    s.flush()
    nearest = s.scalar(
        select(ChunkEmbedding.chunk_id)
        .where(ChunkEmbedding.model_version_id == m.id)
        .order_by(ChunkEmbedding.embedding.cosine_distance([0.1, 0.2, 0.3]))
        .limit(1)
    )
    assert nearest == c.id
    rejects(
        s,
        ChunkEmbedding(
            chunk_id=c.id, model_version_id=m.id, embedding=[1, 0, 0], source_text_sha256=HASH
        ),
    )
    rejects(
        s,
        ChunkEmbedding(
            chunk_id=second.id, model_version_id=m4.id, embedding=[1, 0, 0], source_text_sha256=HASH
        ),
    )
    rejects(
        s,
        ChunkEmbedding(
            chunk_id=second.id, model_version_id=mp.id, embedding=[1, 0, 0], source_text_sha256=HASH
        ),
    )
    with pytest.raises(IntegrityError), s.begin_nested():
        s.execute(delete(ModelVersion).where(ModelVersion.id == m.id))
    with pytest.raises(IntegrityError), s.begin_nested():
        s.execute(update(ModelVersion).where(ModelVersion.id == m.id).values(embedding_dim=2))
    s.execute(delete(Paper).where(Paper.id == p.id))
    s.flush()
    for cls in (PaperAuthor, PaperFile, PaperSection, Chunk, ChunkEmbedding, PaperEmbedding):
        assert s.scalar(select(func.count()).select_from(cls)) == 0
    s.refresh(run)
    assert run.paper_file_id is None and run.status == "failed"


def test_section_cascade_and_optional_section(session):
    p, f, sec, c = paper_tree(session)
    session.execute(delete(PaperSection).where(PaperSection.id == sec.id))
    assert session.scalar(select(func.count()).select_from(Chunk)) == 0
    session.add(
        Chunk(
            paper_id=p.id,
            chunk_index=0,
            chunk_type="body",
            content="No section",
            content_sha256=HASH,
        )
    )
    session.flush()


def test_session_rollback(db_engine):
    with pytest.raises(RuntimeError):
        with session_scope(db_engine) as s:
            s.add(Paper(title="Must roll back"))
            s.flush()
            raise RuntimeError("Test rollback")
    with db_engine.connect() as conn:
        assert conn.scalar(text("SELECT count(*) FROM papers")) == 0


def test_readiness(db_engine, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", db_engine.url.render_as_string(hide_password=False))
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200
        monkeypatch.setenv(
            "DATABASE_URL", db_engine.url.set(port=1).render_as_string(hide_password=False)
        )
        assert client.get("/ready").status_code == 503
