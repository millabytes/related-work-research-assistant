"""Orchestrate chunk and sentence evidence without persisting request results."""

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import Engine

from related_work_assistant.retrieval.chunks import (
    ChunkPair,
    query_bge_model_version,
    retrieve_chunk_pairs,
)
from related_work_assistant.retrieval.reranking import (
    get_chunk_reranker,
    register_reranker_version,
    rerank_chunk_pairs,
)
from related_work_assistant.retrieval.sentences import (
    SentencePair,
    get_sentence_encoder,
    rerank_sentence_pairs,
    sentence_candidates,
)


@dataclass(frozen=True)
class EvidenceBundle:
    bge_model_version_id: UUID
    bge_checkpoint_revision: str
    reranker_model_version_id: UUID
    bge_chunk_candidates: list[ChunkPair]
    reranked_chunk_pairs: list[ChunkPair]
    bge_sentence_candidates: list[SentencePair]
    sentence_pairs: list[SentencePair]
    warnings: list[str]


def retrieve_evidence(
    engine: Engine,
    query_paper_id: UUID,
    recommended_paper_id: UUID,
    *,
    chunk_candidate_count: int = 20,
    chunk_pair_count: int = 3,
    sentence_candidate_count: int = 20,
    sentence_pair_count: int = 3,
) -> EvidenceBundle:
    candidates = retrieve_chunk_pairs(
        engine,
        query_paper_id,
        [recommended_paper_id],
        pairs_per_paper=max(chunk_candidate_count, chunk_pair_count),
    )[recommended_paper_id]
    reranker = get_chunk_reranker()
    reranker_model_version_id = register_reranker_version(engine, reranker.descriptor)
    reranked = rerank_chunk_pairs(
        {recommended_paper_id: candidates},
        reranker,
        final_pairs_per_paper=chunk_pair_count,
    )[recommended_paper_id]
    bge_version = query_bge_model_version(engine, query_paper_id)
    sentence_encoder = get_sentence_encoder(bge_version.checkpoint_revision)
    sentence_bge = sentence_candidates(
        reranked,
        sentence_encoder,
        candidate_count=sentence_candidate_count,
    )
    final_sentences = rerank_sentence_pairs(
        sentence_bge,
        reranker,
        final_count=sentence_pair_count,
    )
    warnings = []
    if not candidates:
        warnings.append("No BGE chunk candidates were available for this paper pair.")
    if not sentence_bge:
        warnings.append(
            "No reliably mapped GROBID sentence pairs were available; no coordinates were inferred."
        )
    if len(final_sentences) < sentence_pair_count:
        warnings.append(
            f"Only {len(final_sentences)} distinct mapped sentence pairs were available."
        )
    return EvidenceBundle(
        bge_model_version_id=bge_version.id,
        bge_checkpoint_revision=bge_version.checkpoint_revision,
        reranker_model_version_id=reranker_model_version_id,
        bge_chunk_candidates=candidates,
        reranked_chunk_pairs=reranked,
        bge_sentence_candidates=sentence_bge,
        sentence_pairs=final_sentences,
        warnings=warnings,
    )
