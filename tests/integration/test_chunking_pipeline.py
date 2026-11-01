from pathlib import Path

import pymupdf
import pytest
from sqlalchemy import func, select

from related_work_assistant.chunking.section_aware import ChunkingConfig
from related_work_assistant.core.config import Settings
from related_work_assistant.database.models import Chunk, PaperFile, PaperSection
from related_work_assistant.database.session import session_scope
from related_work_assistant.extraction.grobid import parse_header
from related_work_assistant.pipelines.chunking import process_paper_file
from related_work_assistant.pipelines.ingestion import ingest_pdf

pytestmark = pytest.mark.integration
HEADER = (Path(__file__).parents[1] / "fixtures/header.tei.xml").read_text()
FULLTEXT = (Path(__file__).parents[1] / "fixtures/fulltext.tei.xml").read_text()


class FakeGrobid:
    def __init__(self):
        self.calls = 0

    def version(self):
        return "test-fixture"

    def extract(self, path):
        self.calls += 1
        return parse_header(HEADER), HEADER

    def extract_fulltext(self, path):
        self.calls += 1
        return FULLTEXT


@pytest.fixture
def ingestion_connection(db_engine):
    with db_engine.connect() as connection:
        transaction = connection.begin()
        try:
            yield connection
        finally:
            transaction.rollback()


def sample_pdf(path: Path) -> Path:
    path.parent.mkdir(parents=True)
    with pymupdf.open() as document:
        first = document.new_page(width=600, height=800)
        first.insert_text((40, 80), "A short abstract sentence. It has a second sentence.")
        first.insert_text((40, 170), "The introduction starts with a mapped sentence.")
        first.insert_text((40, 185), "The next sentence continues the same paragraph.")
        first.insert_text((40, 230), "A separate paragraph should retain its logical boundary.")
        second = document.new_page(width=600, height=800)
        second.insert_text(
            (40, 110), "The method is evaluated independently from the introduction."
        )
        document.save(path)
    return path


def test_chunking_pipeline_cache_idempotence_and_config_change(ingestion_connection, tmp_path):
    settings = Settings(_env_file=None, postgres_password="test", storage_root=tmp_path)
    pdf = sample_pdf(tmp_path / "data/raw_data/papers/paper.pdf")
    grobid = FakeGrobid()
    ingested = ingest_pdf(pdf, settings, ingestion_connection, grobid)
    with session_scope(ingestion_connection) as session:
        file_id = session.scalar(select(PaperFile.id))

    config = ChunkingConfig(min_tokens=5, target_tokens=12, max_tokens=20)
    first = process_paper_file(file_id, settings, ingestion_connection, grobid, config)
    assert first["status"] == "succeeded"
    assert first["section_count"] == 3
    assert first["chunk_count"] >= 3
    assert (tmp_path / first["artifact_dir"] / "fulltext.tei.xml").exists()
    assert (tmp_path / first["artifact_dir"] / "chunking.json").exists()

    second = process_paper_file(file_id, settings, ingestion_connection, grobid, config)
    assert second["status"] == "skipped"
    assert second["fulltext_cache_hit"]
    with session_scope(ingestion_connection) as session:
        original_ids = list(session.scalars(select(Chunk.id).order_by(Chunk.chunk_index)))
        assert session.scalar(select(func.count()).select_from(PaperSection)) == 3
        assert all(
            section.source_file_id == file_id for section in session.scalars(select(PaperSection))
        )
        assert all(
            "reference" not in text.casefold() for text in session.scalars(select(Chunk.content))
        )

    changed = ChunkingConfig(min_tokens=5, target_tokens=8, max_tokens=12)
    third = process_paper_file(file_id, settings, ingestion_connection, grobid, changed)
    assert third["status"] == "succeeded"
    with session_scope(ingestion_connection) as session:
        replacement_ids = list(session.scalars(select(Chunk.id).order_by(Chunk.chunk_index)))
        assert set(original_ids).isdisjoint(replacement_ids)
        assert session.scalar(select(func.count()).select_from(PaperSection)) == 3
        assert {version for version in session.scalars(select(Chunk.chunker_version))} == {
            changed.version
        }

    assert ingested["paper_id"] == third["paper_id"]
