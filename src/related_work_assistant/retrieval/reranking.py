"""Lazy cross-encoder reranking for BGE chunk-pair candidates."""

import json
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from threading import Lock
from uuid import UUID

import numpy as np
from sqlalchemy import Engine, select

from related_work_assistant.core.config import Settings
from related_work_assistant.database.models import ModelVersion
from related_work_assistant.database.session import session_scope
from related_work_assistant.embeddings.encoders import (
    MAX_MODEL_PARAMETERS,
    Device,
    choose_device,
    resolve_revision,
)
from related_work_assistant.retrieval.chunks import ChunkPair

RERANKER_REPOSITORY = "cross-encoder/ms-marco-MiniLM-L6-v2"


@dataclass(frozen=True)
class RerankerDescriptor:
    repository_id: str
    checkpoint_revision: str
    parameter_count: int
    config: dict[str, object]


class ChunkPairReranker:
    """Thread-safe wrapper around the MS MARCO MiniLM cross-encoder."""

    repository_id = RERANKER_REPOSITORY

    def __init__(
        self,
        cache_dir: Path,
        *,
        device: Device = "auto",
        revision: str | None = None,
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.device = choose_device(device)
        self.checkpoint_revision = resolve_revision(self.repository_id, revision)
        self.max_length = max_length
        self.batch_size = batch_size
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = CrossEncoder(
            self.repository_id,
            revision=self.checkpoint_revision,
            cache_folder=str(cache_dir),
            device=self.device,
            max_length=self.max_length,
        )
        self.parameter_count = sum(parameter.numel() for parameter in self.model.model.parameters())
        if self.parameter_count >= MAX_MODEL_PARAMETERS:
            raise ValueError("Cross-encoder exceeds the local model-size constraint")
        self._inference_lock = Lock()

    @property
    def descriptor(self) -> RerankerDescriptor:
        return RerankerDescriptor(
            repository_id=self.repository_id,
            checkpoint_revision=self.checkpoint_revision,
            parameter_count=self.parameter_count,
            config={
                "library": "sentence-transformers",
                "input": "(query_chunk_text, recommended_chunk_text)",
                "max_length": self.max_length,
                "truncation": "tokenizer longest_first",
                "score": "raw sequence-classification logit",
                "batch_size": self.batch_size,
            },
        )

    def score(self, text_pairs: list[tuple[str, str]]) -> np.ndarray:
        import torch

        if not text_pairs:
            return np.empty(0, dtype=np.float32)
        with self._inference_lock:
            values = self.model.predict(
                text_pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                activation_fn=torch.nn.Identity(),
                convert_to_numpy=True,
            )
        scores = np.asarray(values, dtype=np.float32).reshape(-1)
        if scores.shape != (len(text_pairs),) or not np.isfinite(scores).all():
            raise ValueError("Cross-encoder returned invalid reranking scores")
        return scores


_singleton: ChunkPairReranker | None = None
_singleton_lock = Lock()


def get_chunk_reranker() -> ChunkPairReranker:
    """Load one reranker per API process and reuse it for later requests."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                settings = Settings()
                _singleton = ChunkPairReranker(
                    settings.model_cache_dir,
                    device=settings.reranker_device,
                    revision=settings.reranker_revision,
                    max_length=settings.reranker_max_length,
                    batch_size=settings.reranker_batch_size,
                )
    return _singleton


def _model_key(descriptor: RerankerDescriptor) -> str:
    identity = {
        "role": "reranking",
        "repository_id": descriptor.repository_id,
        "checkpoint_revision": descriptor.checkpoint_revision,
        "config": descriptor.config,
    }
    digest = sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return f"reranking:{descriptor.repository_id}:{digest[:16]}"


def register_reranker_version(engine: Engine, descriptor: RerankerDescriptor) -> UUID:
    """Persist model provenance, never request-specific scores or rankings."""
    key = _model_key(descriptor)
    with session_scope(engine) as session:
        row = session.scalar(select(ModelVersion).where(ModelVersion.model_key == key))
        if row is None:
            row = ModelVersion(
                model_key=key,
                role="reranking",
                provider="huggingface",
                repository_id=descriptor.repository_id,
                checkpoint_revision=descriptor.checkpoint_revision,
                parameter_count=descriptor.parameter_count,
                config=descriptor.config,
            )
            session.add(row)
            session.flush()
        return row.id


def rerank_chunk_pairs(
    candidates: dict[UUID, list[ChunkPair]],
    reranker: ChunkPairReranker,
    *,
    final_pairs_per_paper: int,
) -> dict[UUID, list[ChunkPair]]:
    indexed: list[tuple[UUID, ChunkPair]] = [
        (paper_id, pair)
        for paper_id, pairs in candidates.items()
        for pair in pairs
    ]
    scores = reranker.score(
        [
            (pair.query_chunk.text, pair.recommended_chunk.text)
            for _paper_id, pair in indexed
        ]
    )
    grouped: dict[UUID, list[ChunkPair]] = {paper_id: [] for paper_id in candidates}
    for (paper_id, pair), score in zip(indexed, scores, strict=True):
        grouped[paper_id].append(replace(pair, reranking_score=float(score), rank=0))
    for paper_id, pairs in grouped.items():
        ordered = sorted(
            pairs,
            key=lambda pair: (
                -float(pair.reranking_score),
                -pair.similarity_score,
                str(pair.query_chunk.chunk_id),
                str(pair.recommended_chunk.chunk_id),
            ),
        )[:final_pairs_per_paper]
        grouped[paper_id] = [replace(pair, rank=rank) for rank, pair in enumerate(ordered, 1)]
    return grouped


def finalize_bge_pairs(
    candidates: dict[UUID, list[ChunkPair]], final_pairs_per_paper: int
) -> dict[UUID, list[ChunkPair]]:
    return {
        paper_id: [replace(pair, rank=rank) for rank, pair in enumerate(pairs[:final_pairs_per_paper], 1)]
        for paper_id, pairs in candidates.items()
    }
