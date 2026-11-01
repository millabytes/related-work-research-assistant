"""Build reproducible PRIMERA and Qwen evidence snapshots."""

import json
import re
from hashlib import sha256
from uuid import UUID

from sqlalchemy import Engine, select

from related_work_assistant.database.models import Paper, PaperAuthor
from related_work_assistant.database.session import session_scope
from related_work_assistant.retrieval.chunks import retrieve_chunk_pairs
from related_work_assistant.retrieval.papers import recommend_papers
from related_work_assistant.retrieval.reranking import (
    get_chunk_reranker,
    register_reranker_version,
    rerank_chunk_pairs,
)


class EvidenceUnavailableError(RuntimeError):
    pass


def canonical_hash(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _citation_key(paper: Paper, authors: list[str]) -> str:
    if paper.citation_key:
        return paper.citation_key
    lead = authors[0].split()[-1] if authors else "paper"
    lead = re.sub(r"[^A-Za-z0-9]", "", lead).lower() or "paper"
    return f"{lead}{paper.publication_year or 'nd'}-{str(paper.id)[:8]}"


def _paper_record(session, paper_id: UUID) -> dict[str, object]:
    paper = session.get(Paper, paper_id)
    if paper is None:
        raise EvidenceUnavailableError(f"Paper {paper_id} was not found")
    authors = list(
        session.scalars(
            select(PaperAuthor.full_name)
            .where(PaperAuthor.paper_id == paper_id)
            .order_by(PaperAuthor.author_position)
        )
    )
    if not paper.abstract:
        raise EvidenceUnavailableError(f"Paper {paper_id} has no abstract")
    return {
        "paper_id": str(paper.id),
        "citation_key": _citation_key(paper, authors),
        "title": paper.title,
        "authors": authors,
        "year": paper.publication_year,
        "venue": paper.venue_name,
        "abstract": paper.abstract,
    }


def build_evidence_snapshot(engine: Engine, paper_id: UUID, model_key: str) -> dict[str, object]:
    recommendations = recommend_papers(engine, paper_id, top_k=5)
    if len(recommendations) != 5:
        raise EvidenceUnavailableError(
            f"Generation requires exactly 5 recommendations; found {len(recommendations)}"
        )
    recommended_ids = [item.paper_id for item in recommendations]
    with session_scope(engine) as session:
        query = _paper_record(session, paper_id)
        papers = [_paper_record(session, item.paper_id) for item in recommendations]
    snapshot: dict[str, object] = {
        "schema_version": "related-work-evidence-v1",
        "model_key": model_key,
        "query_paper": query,
        "recommended_papers": papers,
        "paper_retrieval": [
            {"paper_id": str(item.paper_id), "rank": item.rank, "score": item.similarity_score}
            for item in recommendations
        ],
    }
    if model_key == "primera":
        snapshot["document_separator"] = "<doc-sep>"
        return snapshot
    if model_key not in ("qwen", "qwen_lora"):
        raise ValueError(f"Unsupported generation model: {model_key}")

    candidates = retrieve_chunk_pairs(engine, paper_id, recommended_ids, pairs_per_paper=20)
    reranker = get_chunk_reranker()
    reranker_version_id = register_reranker_version(engine, reranker.descriptor)
    reranked = rerank_chunk_pairs(candidates, reranker, final_pairs_per_paper=3)
    chunk_catalog: dict[str, dict[str, object]] = {}
    pair_records: list[dict[str, object]] = []
    for recommended_id in recommended_ids:
        pairs = reranked.get(recommended_id, [])
        if len(pairs) != 3:
            raise EvidenceUnavailableError(
                f"Recommended paper {recommended_id} has {len(pairs)} reranked pairs; 3 required"
            )
        for pair in pairs:
            ids = []
            for side, chunk in (
                ("query", pair.query_chunk),
                ("recommended", pair.recommended_chunk),
            ):
                chunk_id = str(chunk.chunk_id)
                ids.append(chunk_id)
                chunk_catalog.setdefault(
                    chunk_id,
                    {
                        "chunk_id": chunk_id,
                        "paper_id": str(paper_id if side == "query" else recommended_id),
                        "text": chunk.text,
                        "section_heading": chunk.section_heading,
                        "section_type": chunk.section_type,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                    },
                )
            pair_records.append(
                {
                    "recommended_paper_id": str(recommended_id),
                    "rank": pair.rank,
                    "query_chunk_id": ids[0],
                    "recommended_chunk_id": ids[1],
                    "bge_score": pair.similarity_score,
                    "reranking_score": pair.reranking_score,
                }
            )
    snapshot["reranker_model_version_id"] = str(reranker_version_id)
    snapshot["chunk_catalog"] = list(chunk_catalog.values())
    snapshot["chunk_pairs"] = pair_records
    return snapshot


def evidence_hash(snapshot: dict[str, object]) -> str:
    return canonical_hash(snapshot)
