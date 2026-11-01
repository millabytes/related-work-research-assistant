from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from related_work_assistant.api.dependencies import get_engine
from related_work_assistant.api.main import app
from related_work_assistant.core.config import Settings
from related_work_assistant.database.models import (
    ModelVersion,
    Paper,
    PaperAuthor,
    PaperEmbedding,
    RelatedWorkGeneration,
)
from related_work_assistant.database.session import session_scope
from related_work_assistant.generation.generators import GenerationOutput
from related_work_assistant.generation.service import bulk_regenerate_qwen, qwen_job_status
from related_work_assistant.generation.worker import claim_job, process_job

pytestmark = pytest.mark.integration
HASH = "f" * 64


def seed_retrieval(engine):
    with session_scope(engine) as session:
        papers = [
            Paper(title="Query", publication_year=2022, venue_name="ICLR"),
            Paper(title="Near", publication_year=2020, venue_name="NeurIPS Conference"),
            Paper(title="Middle", publication_year=2021, venue_name="ICML"),
            Paper(title="Far", publication_year=2018, venue_name="NeurIPS Workshop"),
            Paper(title="No embedding", publication_year=2020, venue_name="ICML"),
            Paper(title="Other model only", publication_year=2020, venue_name="ICML"),
        ]
        session.add_all(papers)
        session.flush()
        for position, name in enumerate(["Ada Author", "Bob Author"], start=1):
            session.add(
                PaperAuthor(
                    paper_id=papers[1].id,
                    author_position=position,
                    full_name=name,
                )
            )
        model = ModelVersion(
            model_key="specter-test-v1",
            role="paper_embedding",
            repository_id="allenai/specter2_base",
            checkpoint_revision="a" * 40,
            adapter_repository_id="allenai/specter2",
            adapter_revision="b" * 40,
            embedding_dim=3,
        )
        other_model = ModelVersion(
            model_key="specter-test-v2",
            role="paper_embedding",
            repository_id="allenai/specter2_base",
            checkpoint_revision="c" * 40,
            adapter_repository_id="allenai/specter2",
            adapter_revision="d" * 40,
            embedding_dim=3,
        )
        session.add_all([model, other_model])
        session.flush()
        vectors = ([1, 0, 0], [0.9, 0.1, 0], [0.7, 0.7, 0], [0, 1, 0])
        session.add_all(
            PaperEmbedding(
                paper_id=paper.id,
                model_version_id=model.id,
                embedding=vector,
                source_text_sha256=HASH,
            )
            for paper, vector in zip(papers[:4], vectors, strict=True)
        )
        session.add(
            PaperEmbedding(
                paper_id=papers[5].id,
                model_version_id=other_model.id,
                embedding=[1, 0, 0],
                source_text_sha256=HASH,
            )
        )
        session.flush()
        return papers, [model.id, other_model.id]


@pytest.fixture
def retrieval_database(db_engine):
    papers, model_ids = seed_retrieval(db_engine)
    try:
        yield db_engine, papers
    finally:
        with session_scope(db_engine) as session:
            session.execute(delete(Paper).where(Paper.id.in_([paper.id for paper in papers])))
            session.execute(delete(ModelVersion).where(ModelVersion.id.in_(model_ids)))


@pytest.fixture
def retrieval_client(retrieval_database, monkeypatch, tmp_path):
    engine, papers = retrieval_database
    monkeypatch.setenv("QWEN_LORA_DIR", str(tmp_path / "missing-qwen-lora-adapter"))

    def override_engine():
        return engine

    app.dependency_overrides[get_engine] = override_engine
    try:
        with TestClient(app) as client:
            yield client, papers
    finally:
        app.dependency_overrides.clear()


def test_ranked_recommendations_exclude_query_and_other_versions(retrieval_client):
    client, papers = retrieval_client
    response = client.get(
        f"/papers/{papers[0].id}/recommendations",
        params={"top_k": 5, "chunk_pairs_per_paper": 0},
    )
    assert response.status_code == 200
    results = response.json()
    assert [item["title"] for item in results] == ["Near", "Middle", "Far"]
    assert [item["rank"] for item in results] == [1, 2, 3]
    assert str(papers[0].id) not in {item["paper_id"] for item in results}
    assert results[0]["authors"] == ["Ada Author", "Bob Author"]
    assert results[0]["similarity_score"] == pytest.approx(0.9938837)
    assert results[1]["similarity_score"] == pytest.approx(0.7071068)
    assert results[2]["similarity_score"] == pytest.approx(0.0)


def test_metadata_filters_apply_before_limit(retrieval_client):
    client, papers = retrieval_client
    response = client.get(
        f"/papers/{papers[0].id}/recommendations",
        params={
            "top_k": 1,
            "year_from": 2019,
            "year_to": 2020,
            "venue": "neurips",
            "chunk_pairs_per_paper": 0,
        },
    )
    assert response.status_code == 200
    assert [item["title"] for item in response.json()] == ["Near"]
    invalid = client.get(
        f"/papers/{papers[0].id}/recommendations",
        params={"year_from": 2021, "year_to": 2020, "chunk_pairs_per_paper": 0},
    )
    assert invalid.status_code == 422


def test_paper_list_and_error_cases(retrieval_client):
    client, papers = retrieval_client
    listed = client.get("/papers", params={"has_embedding": True}).json()
    assert {item["title"] for item in listed} == {
        "Query",
        "Near",
        "Middle",
        "Far",
        "Other model only",
    }
    assert all(item["has_specter2_embedding"] for item in listed)
    missing = client.get(f"/papers/{papers[4].id}/recommendations")
    assert missing.status_code == 409
    assert "no SPECTER2" in missing.json()["detail"]
    assert client.get(f"/papers/{uuid4()}/recommendations").status_code == 404
    no_chunks = client.get(
        f"/papers/{papers[0].id}/recommendations",
        params={"top_k": 1, "chunk_pairs_per_paper": 1},
    )
    assert no_chunks.status_code == 409
    assert "no BGE-small" in no_chunks.json()["detail"]
    missing_pdf = client.get(f"/papers/{papers[0].id}/pdf")
    assert missing_pdf.status_code == 404
    assert "no PDF" in missing_pdf.json()["detail"]


def test_related_work_api_enqueues_durable_idempotent_jobs(retrieval_client):
    client, papers = retrieval_client
    paper_id = papers[0].id
    first = client.post(
        f"/papers/{paper_id}/related-work/generate",
        json={"models": ["primera"]},
    )
    assert first.status_code == 202
    assert len(first.json()["job_ids"]) == 1
    duplicate = client.post(
        f"/papers/{paper_id}/related-work/generate",
        json={"models": ["primera"]},
    )
    assert duplicate.status_code == 202
    assert duplicate.json()["job_ids"] == []
    status_response = client.get(f"/papers/{paper_id}/related-work")
    assert status_response.status_code == 200
    models = {item["model_key"]: item for item in status_response.json()["models"]}
    assert models["primera"]["status"] == "pending"
    assert models["qwen_lora"]["status"] == "not_available"
    unavailable = client.post(
        f"/papers/{paper_id}/related-work/generate",
        json={"models": ["qwen_lora"]},
    )
    assert unavailable.status_code == 422
    assert "unavailable" in unavailable.json()["detail"]
    regenerate = client.post(
        f"/papers/{paper_id}/related-work/regenerate",
        json={"models": ["primera"]},
    )
    assert regenerate.status_code == 202
    assert regenerate.json()["job_ids"] == []


def test_bulk_qwen_enqueue_counts_all_papers(retrieval_client, monkeypatch):
    client, papers = retrieval_client
    engine = client.app.dependency_overrides[get_engine]()
    calls = []

    def fake_enqueue(_engine, _settings, paper_id, model_keys, *, force):
        calls.append((paper_id, model_keys, force))
        return [uuid4()]

    monkeypatch.setattr("related_work_assistant.generation.service.enqueue", fake_enqueue)
    result = bulk_regenerate_qwen(engine, Settings())
    assert result == {
        "papers": len(papers),
        "queued": len(papers),
        "already_active": 0,
        "failed_to_queue": 0,
        "failures": [],
    }
    assert all(model_keys == ["qwen"] and force for _, model_keys, force in calls)
    status = qwen_job_status(engine)
    assert status["model"] == "qwen"
    assert status["jobs"]["queued"] == 0


def test_worker_persists_caches_and_regenerates_history(retrieval_client, monkeypatch):
    client, papers = retrieval_client
    engine = client.app.dependency_overrides[get_engine]()
    paper_id = papers[0].id
    snapshot = {
        "query_paper": {"paper_id": str(paper_id), "citation_key": "query"},
        "recommended_papers": [],
        "paper_retrieval": [],
    }

    class FakeGenerator:
        model_key = "primera"
        repository_id = "test/primera"
        revision = "9" * 40
        parameter_count = 10

        def __init__(self, **_kwargs):
            pass

        def generate(self, _snapshot):
            return GenerationOutput(
                status="completed",
                text="Grounded related work.",
                input_tokens=100,
                output_tokens=20,
                context_limit=4096,
                input_hash="8" * 64,
                citations=[],
            )

    monkeypatch.setattr(
        "related_work_assistant.generation.worker.build_evidence_snapshot",
        lambda *_args: snapshot,
    )
    monkeypatch.setattr("related_work_assistant.generation.worker.PrimeraGenerator", FakeGenerator)

    assert (
        client.post(
            f"/papers/{paper_id}/related-work/generate", json={"models": ["primera"]}
        ).status_code
        == 202
    )
    first_job = claim_job(engine)
    process_job(engine, Settings(), first_job)
    status_payload = client.get(f"/papers/{paper_id}/related-work?model=primera").json()
    assert status_payload["models"][0]["status"] == "completed"
    assert len(status_payload["models"][0]["history"]) == 1
    assert status_payload["models"][0]["latest"]["evidence_snapshot"] == snapshot
    assert "rendered_model_input" in status_payload["models"][0]["latest"]
    assert "evidence_snapshot" not in status_payload["models"][0]["history"][0]

    client.post(f"/papers/{paper_id}/related-work/generate", json={"models": ["primera"]})
    cached_job = claim_job(engine)
    process_job(engine, Settings(), cached_job)
    with session_scope(engine) as session:
        assert session.query(RelatedWorkGeneration).filter_by(paper_id=paper_id).count() == 1

    client.post(f"/papers/{paper_id}/related-work/regenerate", json={"models": ["primera"]})
    regenerated_job = claim_job(engine)
    process_job(engine, Settings(), regenerated_job)
    status_payload = client.get(f"/papers/{paper_id}/related-work?model=primera").json()
    assert len(status_payload["models"][0]["history"]) == 2
