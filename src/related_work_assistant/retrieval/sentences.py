"""Sentence evidence extraction, BGE matching, and MiniLM reranking."""

import math
from dataclasses import dataclass, replace
from threading import Lock
from typing import Protocol
from uuid import UUID

import numpy as np

from related_work_assistant.core.config import Settings
from related_work_assistant.embeddings.encoders import BgeChunkEncoder
from related_work_assistant.retrieval.chunks import (
    ChunkPair,
    ChunkPassage,
    near_duplicate,
)


@dataclass(frozen=True)
class SentenceRegion:
    page: int
    bbox: tuple[float, float, float, float]
    coordinate_system: str


@dataclass(frozen=True)
class SentenceEvidence:
    sentence_id: str | None
    chunk_id: UUID
    text: str
    section_heading: str | None
    section_type: str | None
    page_start: int
    page_end: int
    coordinates: list[SentenceRegion]
    mapping_status: str
    text_similarity: float | None


@dataclass(frozen=True)
class SentencePair:
    query_sentence: SentenceEvidence
    recommended_sentence: SentenceEvidence
    bge_similarity_score: float
    reranking_score: float | None
    parent_chunk_pair_rank: int
    rank: int


class SentenceEncoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class PairScorer(Protocol):
    def score(self, text_pairs: list[tuple[str, str]]) -> np.ndarray: ...


class SentenceBgeEncoder:
    def __init__(self, revision: str) -> None:
        settings = Settings()
        self.encoder = BgeChunkEncoder(
            settings.model_cache_dir,
            device=settings.reranker_device,
            revision=revision,
        )
        self.batch_size = settings.reranker_batch_size
        self._lock = Lock()

    def encode(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            return self.encoder.encode(texts, batch_size=self.batch_size)


_encoders: dict[str, SentenceBgeEncoder] = {}
_encoders_lock = Lock()


def get_sentence_encoder(revision: str) -> SentenceBgeEncoder:
    if revision not in _encoders:
        with _encoders_lock:
            if revision not in _encoders:
                _encoders[revision] = SentenceBgeEncoder(revision)
    return _encoders[revision]


def _valid_regions(mapping: dict) -> list[SentenceRegion]:
    if mapping.get("mapping_status") != "mapped":
        return []
    regions = []
    for raw in mapping.get("pymupdf_regions") or []:
        bbox = raw.get("pymupdf_bbox")
        page = raw.get("page")
        if not isinstance(page, int) or page < 1 or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            values = tuple(float(value) for value in bbox)
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in values):
            continue
        x0, y0, x1, y1 = values
        if x1 <= x0 or y1 <= y0:
            continue
        regions.append(
            SentenceRegion(
                page=page,
                bbox=values,
                coordinate_system=str(
                    raw.get("coordinate_system") or "pdf_points_top_left_unrotated"
                ),
            )
        )
    return regions


def extract_mapped_sentences(passage: ChunkPassage) -> list[SentenceEvidence]:
    result = []
    seen = set()
    for mapping in passage.source_mapping:
        text = mapping.get("grobid_text")
        regions = _valid_regions(mapping)
        if not isinstance(text, str) or not text.strip() or not regions:
            continue
        sentence_id = mapping.get("grobid_sentence_id")
        key = (sentence_id, text.strip(), tuple((region.page, region.bbox) for region in regions))
        if key in seen:
            continue
        seen.add(key)
        pages = [region.page for region in regions]
        similarity = mapping.get("text_similarity")
        result.append(
            SentenceEvidence(
                sentence_id=str(sentence_id) if sentence_id else None,
                chunk_id=passage.chunk_id,
                text=text.strip(),
                section_heading=passage.section_heading,
                section_type=passage.section_type,
                page_start=min(pages),
                page_end=max(pages),
                coordinates=regions,
                mapping_status="mapped",
                text_similarity=float(similarity) if isinstance(similarity, (int, float)) else None,
            )
        )
    return result


def _sentence_key(sentence: SentenceEvidence) -> tuple:
    return (sentence.chunk_id, sentence.sentence_id, sentence.text)


def _deduplicate_pairs(pairs: list[SentencePair], limit: int) -> list[SentencePair]:
    selected = []
    for pair in pairs:
        if any(
            _sentence_key(pair.query_sentence) == _sentence_key(existing.query_sentence)
            or _sentence_key(pair.recommended_sentence)
            == _sentence_key(existing.recommended_sentence)
            or near_duplicate(pair.query_sentence.text, existing.query_sentence.text)
            or near_duplicate(
                pair.recommended_sentence.text, existing.recommended_sentence.text
            )
            for existing in selected
        ):
            continue
        selected.append(replace(pair, rank=len(selected) + 1))
        if len(selected) == limit:
            break
    return selected


def sentence_candidates(
    chunk_pairs: list[ChunkPair],
    encoder: SentenceEncoder,
    *,
    candidate_count: int = 20,
) -> list[SentencePair]:
    parent_pairs = []
    unique: dict[tuple, SentenceEvidence] = {}
    for chunk_pair in chunk_pairs:
        query_sentences = extract_mapped_sentences(chunk_pair.query_chunk)
        recommended_sentences = extract_mapped_sentences(chunk_pair.recommended_chunk)
        for sentence in [*query_sentences, *recommended_sentences]:
            unique[_sentence_key(sentence)] = sentence
        for query_sentence in query_sentences:
            for recommended_sentence in recommended_sentences:
                parent_pairs.append(
                    (query_sentence, recommended_sentence, chunk_pair.rank)
                )
    if not parent_pairs:
        return []
    sentences = list(unique.values())
    vectors = encoder.encode([sentence.text for sentence in sentences])
    vector_by_key = {
        _sentence_key(sentence): vector
        for sentence, vector in zip(sentences, vectors, strict=True)
    }
    ranked = sorted(
        (
            SentencePair(
                query_sentence=query_sentence,
                recommended_sentence=recommended_sentence,
                bge_similarity_score=float(
                    np.dot(
                        vector_by_key[_sentence_key(query_sentence)],
                        vector_by_key[_sentence_key(recommended_sentence)],
                    )
                ),
                reranking_score=None,
                parent_chunk_pair_rank=parent_rank,
                rank=0,
            )
            for query_sentence, recommended_sentence, parent_rank in parent_pairs
        ),
        key=lambda pair: (
            -pair.bge_similarity_score,
            pair.parent_chunk_pair_rank,
            pair.query_sentence.text,
            pair.recommended_sentence.text,
        ),
    )
    return _deduplicate_pairs(ranked, candidate_count)


def rerank_sentence_pairs(
    candidates: list[SentencePair],
    scorer: PairScorer,
    *,
    final_count: int = 3,
) -> list[SentencePair]:
    if not candidates:
        return []
    scores = scorer.score(
        [(pair.query_sentence.text, pair.recommended_sentence.text) for pair in candidates]
    )
    scored = [
        replace(pair, reranking_score=float(score), rank=0)
        for pair, score in zip(candidates, scores, strict=True)
    ]
    scored.sort(
        key=lambda pair: (
            -float(pair.reranking_score),
            -pair.bge_similarity_score,
            pair.query_sentence.text,
            pair.recommended_sentence.text,
        )
    )
    return [replace(pair, rank=rank) for rank, pair in enumerate(scored[:final_count], 1)]
