"""Exact SPECTER2 paper-to-paper retrieval."""

from collections import defaultdict
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import Engine, exists, select

from related_work_assistant.database.models import (
    ModelVersion,
    Paper,
    PaperAuthor,
    PaperEmbedding,
)
from related_work_assistant.database.session import session_scope

SPECTER2_BASE = "allenai/specter2_base"
SPECTER2_RETRIEVAL_ADAPTER = "allenai/specter2"


class PaperNotFoundError(LookupError):
    pass


class PaperEmbeddingNotFoundError(LookupError):
    pass


class InvalidPaperFilterError(ValueError):
    pass


@dataclass(frozen=True)
class PaperListItem:
    paper_id: UUID
    title: str | None
    authors: list[str]
    year: int | None
    venue: str | None
    doi: str | None
    has_specter2_embedding: bool


@dataclass(frozen=True)
class PaperRecommendation:
    paper_id: UUID
    title: str | None
    authors: list[str]
    year: int | None
    venue: str | None
    doi: str | None
    similarity_score: float
    rank: int


def validate_filters(year_from: int | None, year_to: int | None, venue: str | None) -> str | None:
    if year_from is not None and year_to is not None and year_from > year_to:
        raise InvalidPaperFilterError("year_from must be less than or equal to year_to")
    if venue is None:
        return None
    normalized = venue.strip()
    if not normalized:
        raise InvalidPaperFilterError("venue must not be blank")
    return normalized


def _literal_contains_pattern(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _authors_by_paper(session, paper_ids: list[UUID]) -> dict[UUID, list[str]]:
    authors: dict[UUID, list[str]] = defaultdict(list)
    if not paper_ids:
        return authors
    rows = session.execute(
        select(PaperAuthor.paper_id, PaperAuthor.full_name)
        .where(PaperAuthor.paper_id.in_(paper_ids))
        .order_by(PaperAuthor.paper_id, PaperAuthor.author_position)
    )
    for paper_id, full_name in rows:
        authors[paper_id].append(full_name)
    return authors


def _is_specter2_version():
    return (
        (ModelVersion.role == "paper_embedding")
        & (ModelVersion.repository_id == SPECTER2_BASE)
        & (ModelVersion.adapter_repository_id == SPECTER2_RETRIEVAL_ADAPTER)
    )


def list_papers(
    engine: Engine,
    *,
    limit: int = 100,
    offset: int = 0,
    has_embedding: bool | None = None,
) -> list[PaperListItem]:
    specter_exists = exists(
        select(PaperEmbedding.id)
        .join(ModelVersion, ModelVersion.id == PaperEmbedding.model_version_id)
        .where(PaperEmbedding.paper_id == Paper.id, _is_specter2_version())
    )
    statement = select(
        Paper.id,
        Paper.title,
        Paper.publication_year,
        Paper.venue_name,
        Paper.doi,
        specter_exists.label("has_embedding"),
    )
    if has_embedding is not None:
        statement = statement.where(specter_exists if has_embedding else ~specter_exists)
    statement = statement.order_by(Paper.title.asc().nulls_last(), Paper.id).offset(offset).limit(limit)

    with session_scope(engine) as session:
        rows = list(session.execute(statement))
        authors = _authors_by_paper(session, [row.id for row in rows])
        return [
            PaperListItem(
                paper_id=row.id,
                title=row.title,
                authors=authors[row.id],
                year=row.publication_year,
                venue=row.venue_name,
                doi=row.doi,
                has_specter2_embedding=row.has_embedding,
            )
            for row in rows
        ]


def recommend_papers(
    engine: Engine,
    paper_id: UUID,
    *,
    top_k: int = 5,
    year_from: int | None = None,
    year_to: int | None = None,
    venue: str | None = None,
) -> list[PaperRecommendation]:
    venue = validate_filters(year_from, year_to, venue)
    with session_scope(engine) as session:
        if session.scalar(select(exists().where(Paper.id == paper_id))) is not True:
            raise PaperNotFoundError(f"Paper {paper_id} was not found")

        query_row = session.execute(
            select(PaperEmbedding.embedding, PaperEmbedding.model_version_id)
            .join(ModelVersion, ModelVersion.id == PaperEmbedding.model_version_id)
            .where(PaperEmbedding.paper_id == paper_id, _is_specter2_version())
            .order_by(ModelVersion.created_at.desc(), ModelVersion.id.desc())
            .limit(1)
        ).one_or_none()
        if query_row is None:
            raise PaperEmbeddingNotFoundError(
                f"Paper {paper_id} has no SPECTER2 retrieval embedding"
            )

        query_vector = list(query_row.embedding)
        distance = PaperEmbedding.embedding.cosine_distance(query_vector)
        statement = (
            select(
                Paper.id,
                Paper.title,
                Paper.publication_year,
                Paper.venue_name,
                Paper.doi,
                (1.0 - distance).label("similarity_score"),
            )
            .join(PaperEmbedding, PaperEmbedding.paper_id == Paper.id)
            .where(
                PaperEmbedding.model_version_id == query_row.model_version_id,
                Paper.id != paper_id,
            )
        )
        if year_from is not None:
            statement = statement.where(Paper.publication_year >= year_from)
        if year_to is not None:
            statement = statement.where(Paper.publication_year <= year_to)
        if venue is not None:
            statement = statement.where(
                Paper.venue_name.ilike(_literal_contains_pattern(venue), escape="\\")
            )
        statement = statement.order_by(distance.asc(), Paper.id.asc()).limit(top_k)
        rows = list(session.execute(statement))
        authors = _authors_by_paper(session, [row.id for row in rows])
        return [
            PaperRecommendation(
                paper_id=row.id,
                title=row.title,
                authors=authors[row.id],
                year=row.publication_year,
                venue=row.venue_name,
                doi=row.doi,
                similarity_score=float(row.similarity_score),
                rank=rank,
            )
            for rank, row in enumerate(rows, start=1)
        ]
