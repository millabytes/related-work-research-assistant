"""PEFT LoRA training for the prepared citation-aware Multi-XScience JSONL files."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from related_work_assistant.core.config import PROJECT_ROOT
from related_work_assistant.datasets.multixscience import QWEN_REPOSITORY
from related_work_assistant.embeddings.encoders import MAX_MODEL_PARAMETERS, resolve_revision
from related_work_assistant.training.lora_artifacts import (
    QWEN_LORA_ROOT,
    QWEN_LORA_SMOKE_ROOT,
    adapter_directory,
    adapter_fingerprint,
    training_artifact_status,
)

DATASET_ROOT = PROJECT_ROOT / "data/datasets/multixscience"
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
MODEL_CACHE = PROJECT_ROOT / "models_cache/huggingface"


@dataclass(frozen=True)
class QwenLoraTrainingConfig:
    output_root: Path = QWEN_LORA_ROOT
    dataset_root: Path = DATASET_ROOT
    base_model_revision: str | None = None
    train_limit: int | None = None
    validation_limit: int | None = None
    max_sequence_tokens: int = 32768
    num_train_epochs: float = 3.0
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_total_limit: int = 2
    seed: int = 42
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = True
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    mixed_precision: Literal["auto", "none", "fp16", "bf16"] = "auto"
    resume_from_checkpoint: str | None = None
    smoke_test: bool = False

    @classmethod
    def smoke(cls, **overrides: Any) -> QwenLoraTrainingConfig:
        values: dict[str, Any] = {
            "output_root": QWEN_LORA_SMOKE_ROOT,
            "train_limit": 8,
            "validation_limit": 4,
            "max_steps": 2,
            "gradient_accumulation_steps": 1,
            "logging_steps": 1,
            "save_total_limit": 1,
            "smoke_test": True,
        }
        values.update(overrides)
        return cls(**values)


class JsonlChatDataset(Dataset):
    """Random-access JSONL dataset with exact assistant-only labels and no truncation."""

    def __init__(self, path: Path, tokenizer: Any, *, limit: int | None, max_tokens: int):
        if not path.is_file():
            raise FileNotFoundError(f"Prepared dataset split is missing: {path}")
        self.path = path
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.offsets: list[int] = []
        with path.open("rb") as stream:
            while limit is None or len(self.offsets) < limit:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            row = json.loads(stream.readline())
        input_messages = row["input_messages"]
        messages = row["messages"]
        prompt_ids = self.tokenizer.apply_chat_template(
            input_messages, tokenize=True, add_generation_prompt=True
        )
        full_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                f"{self.path.name}:{row['split_index']} has an unstable assistant boundary"
            )
        if len(full_ids) > self.max_tokens:
            raise ValueError(
                f"{self.path.name}:{row['split_index']} has {len(full_ids)} tokens; "
                f"limit is {self.max_tokens}; truncation is disabled"
            )
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        if not any(label != -100 for label in labels):
            raise ValueError(f"{self.path.name}:{row['split_index']} has no assistant labels")
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


class AssistantOnlyCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(feature["input_ids"]) for feature in features)
        batch_size = len(features)
        input_ids = torch.full((batch_size, width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, width), dtype=torch.long)
        labels = torch.full((batch_size, width), -100, dtype=torch.long)
        for row, feature in enumerate(features):
            length = len(feature["input_ids"])
            input_ids[row, :length] = torch.tensor(feature["input_ids"], dtype=torch.long)
            attention_mask[row, :length] = 1
            labels[row, :length] = torch.tensor(feature["labels"], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
        json.dump(value, file, ensure_ascii=False, indent=2, default=str)
        file.write("\n")
        temporary = Path(file.name)
    os.replace(temporary, path)


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _precision_flags(config: QwenLoraTrainingConfig) -> tuple[bool, bool]:
    if config.mixed_precision == "none":
        return False, False
    if config.mixed_precision == "fp16":
        return True, False
    if config.mixed_precision == "bf16":
        return False, True
    if torch.cuda.is_available():
        return (False, True) if torch.cuda.is_bf16_supported() else (True, False)
    return False, False


def _validate_device(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")


def _serializable_config(config: QwenLoraTrainingConfig) -> dict[str, Any]:
    value = asdict(config)
    value["output_root"] = str(config.output_root)
    value["dataset_root"] = str(config.dataset_root)
    return value


def resolve_training_revision(repository_id: str, requested: str | None) -> str:
    """Prefer the Hub, but preserve an immutable SHA when working from a local cache."""
    try:
        return resolve_revision(repository_id, requested)
    except Exception:
        repository_cache = MODEL_CACHE / f"models--{repository_id.replace('/', '--')}"
        if requested and len(requested) == 40:
            revision = requested
        else:
            reference = repository_cache / "refs" / (requested or "main")
            if not reference.is_file():
                raise
            revision = reference.read_text(encoding="utf-8").strip()
        if len(revision) != 40 or not (repository_cache / "snapshots" / revision).is_dir():
            raise
        return revision


def run_qwen_lora_training(config: QwenLoraTrainingConfig) -> dict[str, Any]:
    """Train and persist a PEFT adapter; the test split is intentionally never opened."""
    _validate_device(config.device)
    config.output_root.mkdir(parents=True, exist_ok=True)
    checkpoints = config.output_root / "checkpoints"
    adapter_path = adapter_directory(config.output_root)
    train_path = config.dataset_root / "train.jsonl"
    validation_path = config.dataset_root / "validation.jsonl"
    resolved_revision = resolve_training_revision(QWEN_REPOSITORY, config.base_model_revision)
    run_config = {
        **_serializable_config(config),
        "base_model_id": QWEN_REPOSITORY,
        "resolved_base_model_revision": resolved_revision,
        "target_modules": list(TARGET_MODULES),
        "train_sha256": _file_hash(train_path),
        "validation_sha256": _file_hash(validation_path),
        "test_split_used": False,
    }
    _atomic_json(config.output_root / "training_config.json", run_config)
    _atomic_json(
        config.output_root / "status.json",
        {"status": "running", "started_at": datetime.now(UTC).isoformat()},
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            QWEN_REPOSITORY,
            revision=resolved_revision,
            cache_dir=MODEL_CACHE,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        train_dataset = JsonlChatDataset(
            train_path,
            tokenizer,
            limit=config.train_limit,
            max_tokens=config.max_sequence_tokens,
        )
        validation_dataset = JsonlChatDataset(
            validation_path,
            tokenizer,
            limit=config.validation_limit,
            max_tokens=config.max_sequence_tokens,
        )
        if not train_dataset or not validation_dataset:
            raise ValueError("Training and validation datasets must both be non-empty")
        base_model = AutoModelForCausalLM.from_pretrained(
            QWEN_REPOSITORY,
            revision=resolved_revision,
            cache_dir=MODEL_CACHE,
        )
        base_model.config.use_cache = False
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(TARGET_MODULES),
            bias="none",
        )
        model = get_peft_model(base_model, lora_config)
        if config.gradient_checkpointing:
            model.enable_input_require_grads()
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        if total_parameters >= MAX_MODEL_PARAMETERS:
            raise ValueError(
                f"Base model plus LoRA has {total_parameters} parameters; must be <500M"
            )
        fp16, bf16 = _precision_flags(config)
        step_strategy = config.max_steps > 0
        training_args = TrainingArguments(
            output_dir=str(checkpoints),
            overwrite_output_dir=False,
            num_train_epochs=config.num_train_epochs,
            max_steps=config.max_steps,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            logging_steps=config.logging_steps,
            eval_strategy="steps" if step_strategy else "epoch",
            save_strategy="steps" if step_strategy else "epoch",
            eval_steps=config.max_steps if step_strategy else None,
            save_steps=config.max_steps if step_strategy else 500,
            save_total_limit=config.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            gradient_checkpointing=config.gradient_checkpointing,
            fp16=fp16,
            bf16=bf16,
            use_cpu=config.device == "cpu",
            use_mps_device=config.device == "mps",
            seed=config.seed,
            data_seed=config.seed,
            remove_unused_columns=False,
            report_to="none",
            save_safetensors=True,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=AssistantOnlyCollator(tokenizer.pad_token_id),
        )
        resume: bool | str = config.resume_from_checkpoint or False
        train_result = trainer.train(resume_from_checkpoint=resume)
        validation_metrics = trainer.evaluate()
        trainer.save_model(str(adapter_path))
        tokenizer.save_pretrained(adapter_path)
        revision = adapter_fingerprint(adapter_path)
        metrics = {
            "train": train_result.metrics,
            "validation": validation_metrics,
            "log_history": trainer.state.log_history,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
        }
        _atomic_json(config.output_root / "metrics.json", metrics)
        manifest = {
            "status": "completed",
            "created_at": datetime.now(UTC).isoformat(),
            "base_model_id": QWEN_REPOSITORY,
            "base_model_revision": resolved_revision,
            "adapter_revision": revision,
            "adapter_path": str(adapter_path),
            "total_parameter_count": total_parameters,
            "trainable_parameter_count": trainable_parameters,
            "under_500m": total_parameters < MAX_MODEL_PARAMETERS,
            "training_config": run_config,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_validation_loss": trainer.state.best_metric,
        }
        _atomic_json(config.output_root / "manifest.json", manifest)
        _atomic_json(
            config.output_root / "status.json",
            {"status": "completed", "finished_at": datetime.now(UTC).isoformat()},
        )
        return manifest
    except Exception as exc:
        _atomic_json(
            config.output_root / "status.json",
            {
                "status": "failed",
                "finished_at": datetime.now(UTC).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise


def qwen_lora_training_status(*, smoke: bool = False) -> dict[str, Any]:
    return training_artifact_status(QWEN_LORA_SMOKE_ROOT if smoke else QWEN_LORA_ROOT)
