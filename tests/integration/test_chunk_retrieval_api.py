import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select

from related_work_assistant.api.dependencies import get_engine
from related_work_assistant.api.main import app
from related_work_assistant.database.models import (
    Chunk,
    ChunkEmbedding,
    ModelVersion,
    Paper,
    PaperEmbedding,
    PaperSection,
)
from related_work_assistant.database.session import session_scope
from related_work_assistant.retrieval.chunks import near_duplicate
from related_work_assistant.retrieval.reranking import RerankerDescriptor

pytestmark = pytest.mark.integration
HASH = "9" * 64


def basis(index: int) -> list[float]:
    vector = [0.0] * 384
    vector[index] = 1.0
    return vector


def sentence_mapping(text: str, page: int, bbox: list[float], sentence_id: str) -> dict:
    return {
        "grobid_sentence_id": sentence_id,
        "grobid_text": text,
        "mapping_status": "mapped",
        "text_similarity": 1.0,
        "pymupdf_regions": [
            {
                "page": page,
                "pymupdf_bbox": bbox,
                "coordinate_system": "pdf_points_top_left_unrotated",
            }
        ],
    }


def seed_chunk_retrieval(engine):
    with session_scope(engine) as session:
        query = Paper(title="Chunk query", abstract="Query abstract")
        recommended = Paper(title="Chunk recommendation", abstract="Recommendation abstract")
        session.add_all([query, recommended])
        session.flush()
        specter = ModelVersion(
            model_key="chunk-retrieval-specter",
            role="paper_embedding",
            repository_id="allenai/specter2_base",
            checkpoint_revision="1" * 40,
            adapter_repository_id="allenai/specter2",
            adapter_revision="2" * 40,
            embedding_dim=3,
        )
        bge = ModelVersion(
            model_key="chunk-retrieval-bge",
            role="chunk_embedding",
            repository_id="BAAI/bge-small-en-v1.5",
            checkpoint_revision="3" * 40,
            embedding_dim=384,
        )
        other_bge = ModelVersion(
            model_key="chunk-retrieval-other-bge",
            role="chunk_embedding",
            repository_id="BAAI/bge-small-en-v1.5",
            checkpoint_revision="4" * 40,
            embedding_dim=384,
        )
        session.add_all([specter, bge, other_bge])
        session.flush()
        session.add_all(
            [
                PaperEmbedding(
                    paper_id=query.id,
                    model_version_id=specter.id,
                    embedding=[1, 0, 0],
                    source_text_sha256=HASH,
                ),
                PaperEmbedding(
                    paper_id=recommended.id,
                    model_version_id=specter.id,
                    embedding=[0.9, 0.1, 0],
                    source_text_sha256=HASH,
                ),
            ]
        )
        query_sections = [
            PaperSection(
                paper_id=query.id,
                section_order=index,
                heading=heading,
                section_type=section_type,
                page_start=index + 1,
                page_end=index + 1,
            )
            for index, (heading, section_type) in enumerate(
                [("Introduction", "introduction"), ("Methods", "methods"), ("Results", "results")]
            )
        ]
        recommended_sections = [
            PaperSection(
                paper_id=recommended.id,
                section_order=index,
                heading=heading,
                section_type=section_type,
                page_start=index + 4,
                page_end=index + 4,
            )
            for index, (heading, section_type) in enumerate(
                [("Background", "introduction"), ("Approach", "methods"), ("Evaluation", "results")]
            )
        ]
        session.add_all([*query_sections, *recommended_sections])
        session.flush()
        query_texts = [
            "Neural models learn latent representations from scientific data without labels.",
            "Neural models learn latent representations from scientific data without labels. ",
            "Our optimization method alternates between two independently trained objectives.",
            "Results improve accuracy across three external benchmark datasets.",
        ]
        recommended_texts = [
            "Latent neural representations are learned from scientific observations without labels.",
            "Latent neural representations are learned from scientific observations without labels. ",
            "The proposed approach alternates optimization of two separate objectives.",
            "Evaluation shows improved accuracy on three external benchmark datasets.",
        ]
        query_chunks = []
        recommended_chunks = []
        for index, text in enumerate(query_texts):
            section_index = max(0, index - 1)
            chunk = Chunk(
                paper_id=query.id,
                section_id=query_sections[section_index].id,
                chunk_index=index,
                chunk_type="body",
                content=text,
                content_sha256=HASH,
                page_start=section_index + 1,
                page_end=section_index + 1,
                source_spans=[
                    sentence_mapping(
                        text,
                        section_index + 1,
                        [10.0, 20.0, 200.0, 32.0],
                        f"query-{index}",
                    )
                ],
            )
            query_chunks.append(chunk)
        for index, text in enumerate(recommended_texts):
            section_index = max(0, index - 1)
            chunk = Chunk(
                paper_id=recommended.id,
                section_id=recommended_sections[section_index].id,
                chunk_index=index,
                chunk_type="body",
                content=text,
                content_sha256=HASH,
                page_start=section_index + 4,
                page_end=section_index + 4,
                source_spans=[
                    sentence_mapping(
                        text,
                        section_index + 4,
                        [12.0, 25.0, 210.0, 37.0],
                        f"recommended-{index}",
                    )
                ],
            )
            recommended_chunks.append(chunk)
        session.add_all([*query_chunks, *recommended_chunks])
        session.flush()
        dimensions = [0, 0, 1, 2]
        session.add_all(
            ChunkEmbedding(
                chunk_id=chunk.id,
                model_version_id=bge.id,
                embedding=basis(dimensions[index]),
                source_text_sha256=HASH,
            )
            for index, chunk in enumerate(query_chunks)
        )
        session.add_all(
            ChunkEmbedding(
                chunk_id=chunk.id,
                model_version_id=bge.id,
                embedding=basis(dimensions[index]),
                source_text_sha256=HASH,
            )
            for index, chunk in enumerate(recommended_chunks)
        )
        session.add(
            ChunkEmbedding(
                chunk_id=recommended_chunks[3].id,
                model_version_id=other_bge.id,
                embedding=basis(0),
                source_text_sha256=HASH,
            )
        )
        session.flush()
        return query, recommended, [specter.id, bge.id, other_bge.id]


@pytest.fixture
def chunk_retrieval_database(db_engine):
    query, recommended, model_ids = seed_chunk_retrieval(db_engine)
    try:
        yield db_engine, query, recommended
    finally:
        with session_scope(db_engine) as session:
            session.execute(delete(Paper).where(Paper.id.in_([query.id, recommended.id])))
            session.execute(
                delete(ModelVersion).where(ModelVersion.repository_id == "test/fake-reranker")
            )
            session.execute(delete(ModelVersion).where(ModelVersion.id.in_(model_ids)))


def test_api_returns_configurable_diverse_chunk_pairs(chunk_retrieval_database):
    engine, query, recommended = chunk_retrieval_database

    def override_engine():
        return engine

    app.dependency_overrides[get_engine] = override_engine
    try:
        with TestClient(app) as client:
            response = client.get(
                f"/papers/{query.id}/recommendations",
                params={"top_k": 1, "chunk_pairs_per_paper": 3, "rerank": False},
            )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    result = response.json()[0]
    assert result["paper_id"] == str(recommended.id)
    assert len(result["chunk_pairs"]) == 3
    assert [pair["rank"] for pair in result["chunk_pairs"]] == [1, 2, 3]
    assert all(pair["similarity_score"] == pytest.approx(1.0) for pair in result["chunk_pairs"])
    assert {pair["query_chunk"]["section_heading"] for pair in result["chunk_pairs"]} == {
        "Introduction",
        "Methods",
        "Results",
    }
    assert {
        pair["recommended_chunk"]["source_mapping"][0]["pymupdf_regions"][0]["page"]
        for pair in result["chunk_pairs"]
    } == {4, 5, 6}
    query_texts = [pair["query_chunk"]["text"] for pair in result["chunk_pairs"]]
    recommended_texts = [pair["recommended_chunk"]["text"] for pair in result["chunk_pairs"]]
    assert all(
        not near_duplicate(query_texts[left], query_texts[right])
        for left in range(3)
        for right in range(left + 1, 3)
    )
    assert all(
        not near_duplicate(recommended_texts[left], recommended_texts[right])
        for left in range(3)
        for right in range(left + 1, 3)
    )


class FakeReranker:
    descriptor = RerankerDescriptor(
        repository_id="test/fake-reranker",
        checkpoint_revision="8" * 40,
        parameter_count=10,
        config={"max_length": 512, "test": True},
    )

    def __init__(self):
        self.calls = 0
        self.candidate_counts = []

    def score(self, text_pairs):
        self.calls += 1
        self.candidate_counts.append(len(text_pairs))
        return np.arange(len(text_pairs), dtype=np.float32)


class FakeSentenceEncoder:
    def encode(self, texts):
        vectors = []
        for index, _text in enumerate(texts):
            vector = np.zeros(8, dtype=np.float32)
            vector[index % 4] = 1.0
            vectors.append(vector)
        return np.stack(vectors)


def test_api_reranks_candidate_pool_and_registers_version(
    chunk_retrieval_database, monkeypatch
):
    engine, query, _recommended = chunk_retrieval_database
    reranker = FakeReranker()
    monkeypatch.setattr(
        "related_work_assistant.api.papers.get_chunk_reranker", lambda: reranker
    )

    def override_engine():
        return engine

    app.dependency_overrides[get_engine] = override_engine
    try:
        with TestClient(app) as client:
            first = client.get(
                f"/papers/{query.id}/recommendations",
                params={
                    "top_k": 1,
                    "chunk_candidates_per_paper": 3,
                    "chunk_pairs_per_paper": 2,
                    "rerank": True,
                },
            )
            second = client.get(
                f"/papers/{query.id}/recommendations",
                params={
                    "top_k": 1,
                    "chunk_candidates_per_paper": 3,
                    "chunk_pairs_per_paper": 2,
                    "rerank": True,
                },
            )
    finally:
        app.dependency_overrides.clear()
    assert first.status_code == second.status_code == 200
    pairs = first.json()[0]["chunk_pairs"]
    assert len(pairs) == 2
    assert [pair["reranking_score"] for pair in pairs] == [2.0, 1.0]
    assert all(pair["similarity_score"] == pytest.approx(1.0) for pair in pairs)
    assert reranker.calls == 2
    assert reranker.candidate_counts == [3, 3]
    with session_scope(engine) as session:
        assert session.scalar(
            select(func.count())
            .select_from(ModelVersion)
            .where(ModelVersion.repository_id == "test/fake-reranker")
        ) == 1


def test_sentence_evidence_endpoint_returns_original_mapped_coordinates(
    chunk_retrieval_database, monkeypatch
):
    engine, query, recommended = chunk_retrieval_database
    reranker = FakeReranker()
    monkeypatch.setattr(
        "related_work_assistant.retrieval.evidence.get_chunk_reranker", lambda: reranker
    )
    monkeypatch.setattr(
        "related_work_assistant.retrieval.evidence.get_sentence_encoder",
        lambda _revision: FakeSentenceEncoder(),
    )

    def override_engine():
        return engine

    app.dependency_overrides[get_engine] = override_engine
    try:
        with TestClient(app) as client:
            response = client.get(
                f"/papers/{query.id}/recommendations/{recommended.id}/evidence",
                params={
                    "chunk_candidates": 3,
                    "chunk_pairs": 3,
                    "sentence_candidates": 3,
                    "sentence_pairs": 3,
                },
            )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["bge_chunk_candidates"]) == 3
    assert len(payload["reranked_chunk_pairs"]) == 3
    assert len(payload["bge_sentence_candidates"]) == 3
    assert len(payload["sentence_pairs"]) == 3
    first = payload["sentence_pairs"][0]
    assert first["query_sentence"]["text"] in {
        "Neural models learn latent representations from scientific data without labels.",
        "Our optimization method alternates between two independently trained objectives.",
        "Results improve accuracy across three external benchmark datasets.",
    }
    assert first["query_sentence"]["mapping_status"] == "mapped"
    assert first["query_sentence"]["coordinates"][0]["bbox"] == [10.0, 20.0, 200.0, 32.0]
    assert first["recommended_sentence"]["coordinates"][0]["page"] in {4, 5, 6}
    assert first["reranking_score"] is not None
