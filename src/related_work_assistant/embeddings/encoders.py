"""Thin, lazy-loading wrappers around the two approved embedding models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

Device = Literal["auto", "cpu", "cuda", "mps"]
MAX_MODEL_PARAMETERS = 500_000_000


@dataclass(frozen=True)
class ModelDescriptor:
    role: Literal["paper_embedding", "chunk_embedding"]
    repository_id: str
    checkpoint_revision: str
    embedding_dim: int
    parameter_count: int
    config: dict[str, object]
    adapter_repository_id: str | None = None
    adapter_revision: str | None = None
    provider: str = "huggingface"
    distance_metric: str = "cosine"


def choose_device(requested: Device = "auto") -> str:
    """Choose an available accelerator, preferring CUDA and then Apple MPS."""
    import torch

    available = {
        "cuda": torch.cuda.is_available(),
        "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "cpu": True,
    }
    if requested == "auto":
        return next(name for name in ("cuda", "mps", "cpu") if available[name])
    if requested not in available:
        raise ValueError(f"Unsupported device: {requested}")
    if not available[requested]:
        raise ValueError(f"Requested device is unavailable: {requested}")
    return requested


def resolve_revision(repository_id: str, revision: str | None = None) -> str:
    """Resolve a branch/tag to an immutable Hugging Face commit SHA."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repository_id, revision=revision)
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not return a revision for {repository_id}")
    return info.sha


def validate_vectors(vectors: np.ndarray, expected_count: int, expected_dim: int) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (expected_count, expected_dim):
        raise ValueError(
            f"Invalid embedding shape {vectors.shape}; expected ({expected_count}, {expected_dim})"
        )
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding contains NaN or infinity")
    if expected_count and np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("Embedding contains a zero vector")
    return vectors


class Specter2PaperEncoder:
    """SPECTER2 base model plus its proximity/retrieval adapter."""

    repository_id = "allenai/specter2_base"
    adapter_repository_id = "allenai/specter2"
    embedding_dim = 768
    max_length = 512

    def __init__(
        self,
        cache_dir: Path,
        device: Device = "auto",
        revision: str | None = None,
        adapter_revision: str | None = None,
    ) -> None:
        import torch
        from adapters import AutoAdapterModel
        from transformers import AutoTokenizer

        self.device = choose_device(device)
        self.checkpoint_revision = resolve_revision(self.repository_id, revision)
        self.adapter_revision = resolve_revision(
            self.adapter_repository_id, adapter_revision
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = str(cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.repository_id,
            revision=self.checkpoint_revision,
            cache_dir=cache,
        )
        self.model = AutoAdapterModel.from_pretrained(
            self.repository_id,
            revision=self.checkpoint_revision,
            cache_dir=cache,
        )
        adapter_name = self.model.load_adapter(
            self.adapter_repository_id,
            version=self.adapter_revision,
            load_as="proximity",
            set_active=True,
            cache_dir=cache,
        )
        # adapters 1.3 may not retain set_active during loading; make activation explicit.
        self.model.set_active_adapters(adapter_name)
        if not self.model.active_adapters:
            raise RuntimeError("SPECTER2 retrieval adapter was loaded but is not active")
        self.model.to(torch.device(self.device))
        self.model.eval()
        if not self.tokenizer.sep_token:
            raise RuntimeError("SPECTER2 tokenizer has no separator token")
        self.separator = self.tokenizer.sep_token
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        if self.parameter_count >= MAX_MODEL_PARAMETERS:
            raise ValueError("SPECTER2 exceeds the local model-size constraint")

    @property
    def descriptor(self) -> ModelDescriptor:
        return ModelDescriptor(
            role="paper_embedding",
            repository_id=self.repository_id,
            checkpoint_revision=self.checkpoint_revision,
            adapter_repository_id=self.adapter_repository_id,
            adapter_revision=self.adapter_revision,
            embedding_dim=self.embedding_dim,
            parameter_count=self.parameter_count,
            config={
                "adapter_name": "proximity",
                "explicit_adapter_activation": True,
                "input": "title + tokenizer.sep_token + abstract",
                "max_length": self.max_length,
                "pooling": "last_hidden_state[:,0,:]",
                "normalized": False,
            },
        )

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        import torch

        batches: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            encoded = self.tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False,
                max_length=self.max_length,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                output = self.model(**encoded)
            batches.append(output.last_hidden_state[:, 0, :].float().cpu().numpy())
        result = np.concatenate(batches) if batches else np.empty((0, self.embedding_dim))
        return validate_vectors(result, len(texts), self.embedding_dim)


class BgeChunkEncoder:
    """BGE-small passage encoder used for section-aware chunks."""

    repository_id = "BAAI/bge-small-en-v1.5"
    embedding_dim = 384

    def __init__(
        self,
        cache_dir: Path,
        device: Device = "auto",
        revision: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.device = choose_device(device)
        self.checkpoint_revision = resolve_revision(self.repository_id, revision)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = SentenceTransformer(
            self.repository_id,
            revision=self.checkpoint_revision,
            cache_folder=str(cache_dir),
            device=self.device,
        )
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        if self.parameter_count >= MAX_MODEL_PARAMETERS:
            raise ValueError("BGE-small exceeds the local model-size constraint")

    @property
    def descriptor(self) -> ModelDescriptor:
        return ModelDescriptor(
            role="chunk_embedding",
            repository_id=self.repository_id,
            checkpoint_revision=self.checkpoint_revision,
            embedding_dim=self.embedding_dim,
            parameter_count=self.parameter_count,
            config={
                "input": "chunk content without query instruction",
                "normalized": True,
                "sentence_transformers": True,
            },
        )

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return validate_vectors(vectors, len(texts), self.embedding_dim)
