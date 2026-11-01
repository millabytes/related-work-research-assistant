from pathlib import Path
from uuid import UUID

import pymupdf
import pytest
from sqlalchemy import func, select

from related_work_assistant.core.config import Settings
from related_work_assistant.database.models import IngestionRun, Paper, PaperAuthor, PaperFile
from related_work_assistant.database.session import session_scope
from related_work_assistant.extraction.grobid import parse_header
from related_work_assistant.pipelines.ingestion import ingest_pdf, load_metadata

pytestmark = pytest.mark.integration
TEI = (Path(__file__).parents[1] / "fixtures/header.tei.xml").read_text()


class FakeGrobid:
    def __init__(self, fail=False, xml=TEI):
        self.fail = fail
        self.xml = xml
        self.calls = 0

    def version(self):
        return "test-fixture"

    def extract(self, path):
        self.calls += 1
        if self.fail:
            raise RuntimeError("Synthetic service failure")
        return parse_header(self.xml), self.xml

    def extract_fulltext(self, path):
        self.calls += 1
        if self.fail:
            raise RuntimeError("Synthetic service failure")
        return (Path(__file__).parents[1] / "fixtures/fulltext.tei.xml").read_text()


@pytest.fixture
def ingestion_connection(db_engine):
    with db_engine.connect() as connection:
        transaction = connection.begin()
        try:
            yield connection
        finally:
            transaction.rollback()


def sample_pdf(tmp_path, name="paper.pdf", content="A scientific paper"):
    path = tmp_path / name
    with pymupdf.open() as doc:
        doc.new_page().insert_text((40, 80), content)
        doc.save(path)
    return path


def test_ingestion_roundtrip_idempotence_and_duplicate_path(ingestion_connection, tmp_path):
    settings = Settings(_env_file=None, postgres_password="test", storage_root=tmp_path)
    path = sample_pdf(tmp_path)
    grobid = FakeGrobid()
    result = ingest_pdf(path, settings, ingestion_connection, grobid)
    assert result["status"] == "succeeded"
    assert (tmp_path / result["artifact_dir"] / "text.json").exists()
    assert (tmp_path / result["artifact_dir"] / "header.tei.xml").read_text() == TEI
    metadata = load_metadata(ingestion_connection, UUID(result["paper_id"]))
    assert metadata == parse_header(TEI)
    assert ingest_pdf(path, settings, ingestion_connection, grobid)["status"] == "skipped"
    duplicate = tmp_path / "duplicate.pdf"
    duplicate.write_bytes(path.read_bytes())
    assert (
        ingest_pdf(duplicate, settings, ingestion_connection, grobid)["paper_id"]
        == result["paper_id"]
    )
    assert grobid.calls == 1
    with session_scope(ingestion_connection) as session:
        assert session.scalar(select(func.count()).select_from(Paper)) == 1
        assert session.scalar(select(func.count()).select_from(PaperFile)) == 1
        assert session.scalar(select(func.count()).select_from(PaperAuthor)) == 2


def test_failed_run_retry_and_unknown_title(ingestion_connection, tmp_path):
    settings = Settings(_env_file=None, postgres_password="test", storage_root=tmp_path)
    path = sample_pdf(tmp_path)
    with pytest.raises(RuntimeError, match="metadata"):
        ingest_pdf(path, settings, ingestion_connection, FakeGrobid(fail=True))
    with session_scope(ingestion_connection) as session:
        run = session.scalar(select(IngestionRun))
        assert run.status == "failed" and run.paper_file_id is None
        assert session.scalar(select(func.count()).select_from(Paper)) == 0
    empty = '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><sourceDesc><biblStruct/></sourceDesc></fileDesc></teiHeader></TEI>'
    result = ingest_pdf(path, settings, ingestion_connection, FakeGrobid(xml=empty))
    assert load_metadata(ingestion_connection, UUID(result["paper_id"])).title is None
    with session_scope(ingestion_connection) as session:
        assert session.scalar(select(func.count()).select_from(IngestionRun)) == 2


def test_doi_identity_and_force_preserve_metadata(ingestion_connection, tmp_path):
    settings = Settings(_env_file=None, postgres_password="test", storage_root=tmp_path)
    first = ingest_pdf(sample_pdf(tmp_path), settings, ingestion_connection, FakeGrobid())
    second_path = sample_pdf(tmp_path, "version2.pdf", "Different PDF version")
    second = ingest_pdf(second_path, settings, ingestion_connection, FakeGrobid())
    assert second["paper_id"] == first["paper_id"]
    changed = FakeGrobid(xml=TEI.replace("Scientific", "Changed"))
    third = ingest_pdf(second_path, settings, ingestion_connection, changed, force=True)
    assert third["run_id"] != second["run_id"]
    assert (
        load_metadata(ingestion_connection, UUID(third["paper_id"])).title == "A Scientific Paper"
    )
    with session_scope(ingestion_connection) as session:
        assert session.scalar(select(func.count()).select_from(PaperFile)) == 2


def test_storage_path_validation(ingestion_connection, tmp_path):
    settings = Settings(_env_file=None, postgres_password="test", storage_root=tmp_path / "nested")
    with pytest.raises(ValueError, match="STORAGE_ROOT"):
        ingest_pdf(sample_pdf(tmp_path), settings, ingestion_connection, FakeGrobid())
