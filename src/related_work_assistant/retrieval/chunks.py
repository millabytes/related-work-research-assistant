"""On-the-fly BGE chunk-pair retrieval for recommended papers."""

import re
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import Engine, select, true
from sqlalchemy.orm import aliased

from related_work_assistant.database.models import (
    Chunk,
    ChunkEmbedding,
    ModelVersion,
    PaperSection,
)
from related_work_assistant.database.session import session_scope

BGE_SMALL = "BAAI/bge-small-en-v1.5"
NEAR_DUPLICATE_THRESHOLD = 0.9


class ChunkEmbeddingNotFoundError(LookupError):
    pass


@dataclass(frozen=True)
class ChunkPassage:
    chunk_id: UUID
    text: str
    section_heading: str | None
    section_type: str | None
    page_start: int | None
    page_end: int | None
    source_mapping: list[dict[str, object]]


@dataclass(frozen=True)
class ChunkPair:
    query_chunk: ChunkPassage
    recommended_chunk: ChunkPassage
    similarity_score: float
    rank: int
    reranking_score: float | None = None


def _word_shingles(text: str, width: int = 5) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.casefold())
    if len(words) < width:
        return {tuple(words)} if words else set()
    return {tuple(words[index : index + width]) for index in range(len(words) - width + 1)}


def near_duplicate(first: str, second: str, threshold: float = NEAR_DUPLICATE_THRESHOLD) -> bool:
    first_shingles = _word_shingles(first)
    second_shingles = _word_shingles(second)
    if not first_shingles or not second_shingles:
        return first.strip().casefold() == second.strip().casefold()
    return (
        len(first_shingles & second_shingles) / len(first_shingles | second_shingles)
        >= threshold
    )


def _is_bge_version():
    return (
        (ModelVersion.role == "chunk_embedding")
        & (ModelVersion.repository_id == BGE_SMALL)
        & (ModelVersion.embedding_dim == 384)
    )


def _select_query_model_version(session, paper_id: UUID) -> UUID:
    model_version_id = session.scalar(
        select(ModelVersion.id)
        .join(ChunkEmbedding, ChunkEmbedding.model_version_id == ModelVersion.id)
        .join(Chunk, Chunk.id == ChunkEmbedding.chunk_id)
        .where(Chunk.paper_id == paper_id, _is_bge_version())
        .group_by(ModelVersion.id)
        .order_by(ModelVersion.created_at.desc(), ModelVersion.id.desc())
        .limit(1)
    )
    if model_version_id is None:
        raise ChunkEmbeddingNotFoundError(f"Paper {paper_id} has no BGE-small chunk embeddings")
    return model_version_id


def query_bge_model_version(engine: Engine, paper_id: UUID) -> ModelVersion:
    with session_scope(engine) as session:
        model_version_id = _select_query_model_version(session, paper_id)
        return session.get(ModelVersion, model_version_id)


def _passage(row, prefix: str) -> ChunkPassage:
    return ChunkPassage(
        chunk_id=getattr(row, f"{prefix}_chunk_id"),
        text=getattr(row, f"{prefix}_text"),
        section_heading=getattr(row, f"{prefix}_section_heading"),
        section_type=getattr(row, f"{prefix}_section_type"),
        page_start=getattr(row, f"{prefix}_page_start"),
        page_end=getattr(row, f"{prefix}_page_end"),
        source_mapping=getattr(row, f"{prefix}_source_mapping") or [],
    )


def _deduplicate(rows, pair_count: int) -> list[ChunkPair]:
    selected: list[ChunkPair] = []
    for row in rows:
        query = _passage(row, "query")
        recommended = _passage(row, "recommended")
        if any(
            query.chunk_id == pair.query_chunk.chunk_id
            or recommended.chunk_id == pair.recommended_chunk.chunk_id
            or near_duplicate(query.text, pair.query_chunk.text)
            or near_duplicate(recommended.text, pair.recommended_chunk.text)
            for pair in selected
        ):
            continue
        selected.append(
            ChunkPair(
                query_chunk=query,
                recommended_chunk=recommended,
                similarity_score=float(row.similarity_score),
                rank=len(selected) + 1,
            )
        )
        if len(selected) == pair_count:
            break
    return selected


def _candidate_pairs(
    session,
    query_paper_id: UUID,
    recommended_paper_id: UUID,
    model_version_id: UUID,
    pair_count: int,
) -> list[ChunkPair]:
    query_chunk = aliased(Chunk, name="query_chunk")
    recommended_chunk = aliased(Chunk, name="recommended_chunk")
    query_embedding = aliased(ChunkEmbedding, name="query_embedding")
    recommended_embedding = aliased(ChunkEmbedding, name="recommended_embedding")
    query_section = aliased(PaperSection, name="query_section")
    recommended_section = aliased(PaperSection, name="recommended_section")
    distance = query_embedding.embedding.cosine_distance(recommended_embedding.embedding)
    statement = (
        select(
            query_chunk.id.label("query_chunk_id"),
            query_chunk.content.label("query_text"),
            query_section.heading.label("query_section_heading"),
            query_section.section_type.label("query_section_type"),
            query_chunk.page_start.label("query_page_start"),
            query_chunk.page_end.label("query_page_end"),
            query_chunk.source_spans.label("query_source_mapping"),
            recommended_chunk.id.label("recommended_chunk_id"),
            recommended_chunk.content.label("recommended_text"),
            recommended_section.heading.label("recommended_section_heading"),
            recommended_section.section_type.label("recommended_section_type"),
            recommended_chunk.page_start.label("recommended_page_start"),
            recommended_chunk.page_end.label("recommended_page_end"),
            recommended_chunk.source_spans.label("recommended_source_mapping"),
            (1.0 - distance).label("similarity_score"),
        )
        .select_from(query_chunk)
        .join(query_embedding, query_embedding.chunk_id == query_chunk.id)
        .join(recommended_chunk, true())
        .join(recommended_embedding, recommended_embedding.chunk_id == recommended_chunk.id)
        .outerjoin(query_section, query_section.id == query_chunk.section_id)
        .outerjoin(recommended_section, recommended_section.id == recommended_chunk.section_id)
        .where(
            query_chunk.paper_id == query_paper_id,
            recommended_chunk.paper_id == recommended_paper_id,
            query_embedding.model_version_id == model_version_id,
            recommended_embedding.model_version_id == model_version_id,
        )
        .order_by(distance.asc(), query_chunk.id.asc(), recommended_chunk.id.asc())
        .limit(max(30, pair_count * 20))
    )
    return _deduplicate(session.execute(statement), pair_count)


def retrieve_chunk_pairs(
    engine: Engine,
    query_paper_id: UUID,
    recommended_paper_ids: list[UUID],
    *,
    pairs_per_paper: int = 3,
) -> dict[UUID, list[ChunkPair]]:
    if not recommended_paper_ids:
        return {}
    if pairs_per_paper < 1:
        return {paper_id: [] for paper_id in recommended_paper_ids}
    with session_scope(engine) as session:
        model_version_id = _select_query_model_version(session, query_paper_id)
        return {
            paper_id: _candidate_pairs(
                session,
                query_paper_id,
                paper_id,
                model_version_id,
                pairs_per_paper,
            )
            for paper_id in recommended_paper_ids
        }
