"""Evaluation A: abstract-only Multi-XScience benchmark inference and ROUGE."""

import gc
import json
import random
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from related_work_assistant.core.config import PROJECT_ROOT
from related_work_assistant.datasets.multixscience import QWEN_REPOSITORY
from related_work_assistant.embeddings.encoders import Device, choose_device, resolve_revision
from related_work_assistant.evaluation.generation.metrics import (
    aggregate_numeric,
    normalize_benchmark_citations,
    rouge_scores,
)
from related_work_assistant.evaluation.generation.storage import new_run_id, save_run
from related_work_assistant.generation.generators import PRIMERA_REPOSITORY
from related_work_assistant.training.lora_artifacts import QWEN_LORA_ROOT, adapter_status

DATASET_PATH = PROJECT_ROOT / "data/datasets/multixscience/test.jsonl"
MODEL_CACHE = PROJECT_ROOT / "models_cache/huggingface"
MODEL_KEYS = ("primera", "qwen", "qwen_lora")


def sample_test_examples(path: Path, *, limit: int | None, seed: int) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    if limit is not None and limit < len(rows):
        indexes = sorted(random.Random(seed).sample(range(len(rows)), limit))
        rows = [rows[index] for index in indexes]
    for row in rows:
        row["benchmark_sample_id"] = f"{row['example_id']}:{row['split_index']}"
    return rows


class _BenchmarkGenerator:
    model_key: str

    def generate(self, row: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _measure(self, callback):
        import torch

        peak = None
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        text, output_tokens, finish_reason = callback()
        latency = (time.perf_counter() - started) * 1000
        if self.device == "cuda":
            peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        return text, output_tokens, finish_reason, latency, peak


class PrimeraBenchmarkGenerator(_BenchmarkGenerator):
    model_key = "primera"

    def __init__(self, *, device: Device, max_new_tokens: int = 256):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.device = choose_device(device)
        self.revision = resolve_revision(PRIMERA_REPOSITORY)
        self.tokenizer = AutoTokenizer.from_pretrained(
            PRIMERA_REPOSITORY, revision=self.revision, cache_dir=MODEL_CACHE
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            PRIMERA_REPOSITORY, revision=self.revision, cache_dir=MODEL_CACHE
        ).to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.inference_config = {
            "source_format": "target_abstract_then_reference_abstracts_with_doc_sep",
            "max_new_tokens": max_new_tokens,
            "num_beams": 4,
            "no_repeat_ngram_size": 3,
            "length_penalty": 2.0,
            "truncation": False,
        }

    def generate(self, row: dict[str, Any]) -> dict[str, Any]:
        documents = [row["target_abstract"], *(ref["abstract"] for ref in row["references"])]
        source = " <doc-sep> ".join(documents) + " <doc-sep>"
        encoded = self.tokenizer(source, truncation=False, return_tensors="pt")
        if int(encoded["input_ids"].shape[-1]) > 4096:
            raise ValueError("PRIMERA benchmark input exceeds 4096 tokens; truncation is disabled")
        docsep = self.tokenizer.convert_tokens_to_ids("<doc-sep>")
        global_attention = self.torch.zeros_like(encoded["input_ids"])
        global_attention[:, 0] = 1
        global_attention[encoded["input_ids"] == docsep] = 1

        def infer():
            with self.torch.inference_mode():
                sequence = self.model.generate(
                    **{key: value.to(self.device) for key, value in encoded.items()},
                    global_attention_mask=global_attention.to(self.device),
                    max_new_tokens=self.max_new_tokens,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    length_penalty=2.0,
                    use_cache=True,
                )[0]
            return (
                self.tokenizer.decode(sequence, skip_special_tokens=True).strip(),
                int(sequence.shape[-1]),
                "eos" if int(sequence[-1]) == self.tokenizer.eos_token_id else "unknown",
            )

        text, output_tokens, finish_reason, latency, peak = self._measure(infer)
        return {
            "text": text,
            "input_tokens": int(encoded["input_ids"].shape[-1]),
            "output_tokens": output_tokens,
            "finish_reason": finish_reason,
            "latency_ms": latency,
            "peak_memory_mb": peak,
        }


class QwenBenchmarkGenerator(_BenchmarkGenerator):
    def __init__(self, *, model_key: str, device: Device, max_new_tokens: int = 384):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_key = model_key
        self.torch = torch
        self.device = choose_device(device)
        adapter = adapter_status(QWEN_LORA_ROOT) if model_key == "qwen_lora" else None
        if adapter is not None and not adapter["available"]:
            raise FileNotFoundError(str(adapter["reason"]))
        self.adapter_revision = adapter.get("adapter_revision") if adapter else None
        self.revision = (
            str(adapter["manifest"]["base_model_revision"])
            if adapter
            else resolve_revision(QWEN_REPOSITORY)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            QWEN_REPOSITORY, revision=self.revision, cache_dir=MODEL_CACHE
        )
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_REPOSITORY, revision=self.revision, cache_dir=MODEL_CACHE
        )
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter["adapter_path"], is_trainable=False)
        self.model = model.to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        self.inference_config = {
            "source_format": "prepared_abstract_only_chat_messages",
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "truncation": False,
        }

    def generate(self, row: dict[str, Any]) -> dict[str, Any]:
        encoded = self.tokenizer.apply_chat_template(
            row["input_messages"],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_tokens = int(encoded["input_ids"].shape[-1])
        if input_tokens + self.max_new_tokens > 32768:
            raise ValueError("Qwen benchmark input exceeds context; truncation is disabled")
        generation_config = deepcopy(self.model.generation_config)
        generation_config.max_new_tokens = self.max_new_tokens
        generation_config.do_sample = False
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None
        generation_config.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        def infer():
            with self.torch.inference_mode():
                sequence = self.model.generate(
                    **{key: value.to(self.device) for key, value in encoded.items()},
                    generation_config=generation_config,
                )[0, input_tokens:]
            eos_ids = self.model.generation_config.eos_token_id or self.tokenizer.eos_token_id
            eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
            return (
                self.tokenizer.decode(sequence, skip_special_tokens=True).strip(),
                int(sequence.shape[-1]),
                "eos" if len(sequence) and int(sequence[-1]) in eos_ids else "length",
            )

        text, output_tokens, finish_reason, latency, peak = self._measure(infer)
        return {
            "text": text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "finish_reason": finish_reason,
            "latency_ms": latency,
            "peak_memory_mb": peak,
        }


def _model_summary(examples: list[dict[str, Any]], model_key: str) -> dict[str, Any]:
    predictions = [example["predictions"][model_key] for example in examples]
    completed = [item for item in predictions if item["status"] == "completed"]
    metrics = aggregate_numeric(
        [item["rouge"] for item in completed], ("rouge1", "rouge2", "rougeL")
    )
    return {
        "status": "completed" if completed else predictions[0]["status"] if predictions else "empty",
        "evaluated": len(completed),
        "failed": sum(item["status"] == "failed" for item in predictions),
        "unavailable": sum(item["status"] == "unavailable" for item in predictions),
        **metrics,
    }


def run_evaluation_a(*, limit: int | None, seed: int, device: Device = "auto") -> dict[str, Any]:
    rows = sample_test_examples(DATASET_PATH, limit=limit, seed=seed)
    run = {
        "run_id": new_run_id("eval-a"),
        "kind": "evaluation_a",
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "config": {
            "limit": limit,
            "seed": seed,
            "device": device,
            "test_split_only": True,
            "benchmark_prompt": "abstract-only-v1",
            "citation_normalization": "@cite_N and [RN] replaced by <cite> for ROUGE only",
        },
        "sample_ids": [row["benchmark_sample_id"] for row in rows],
        "models": {},
        "examples": [
            {
                "sample_id": row["benchmark_sample_id"],
                "example_id": row["example_id"],
                "split_index": row["split_index"],
                "ground_truth": row["assistant_target"],
                "ground_truth_original": row["original_related_work"],
                "ground_truth_normalized": normalize_benchmark_citations(row["assistant_target"]),
                "predictions": {},
            }
            for row in rows
        ],
    }
    save_run("evaluation_a", run)
    factories = {
        "primera": lambda: PrimeraBenchmarkGenerator(device=device),
        "qwen": lambda: QwenBenchmarkGenerator(model_key="qwen", device=device),
        "qwen_lora": lambda: QwenBenchmarkGenerator(model_key="qwen_lora", device=device),
    }
    for model_key in MODEL_KEYS:
        generator = None
        try:
            generator = factories[model_key]()
            run["models"][model_key] = {
                "status": "running",
                "model_revision": generator.revision,
                "adapter_revision": getattr(generator, "adapter_revision", None),
                "adapter_fingerprint": getattr(generator, "adapter_revision", None),
                "inference_config": generator.inference_config,
            }
        except FileNotFoundError as exc:
            run["models"][model_key] = {"status": "unavailable", "error": str(exc)}
        except Exception as exc:
            run["models"][model_key] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        for row, example in zip(rows, run["examples"], strict=True):
            if generator is None:
                example["predictions"][model_key] = {
                    "status": run["models"][model_key]["status"],
                    "text": None,
                    "rouge": None,
                    "error": run["models"][model_key].get("error"),
                }
                continue
            try:
                prediction = generator.generate(row)
                prediction.update(
                    {
                        "status": "completed",
                        "text_original": prediction["text"],
                        "text_normalized": normalize_benchmark_citations(prediction["text"]),
                        "rouge": rouge_scores(prediction["text"], row["assistant_target"]),
                    }
                )
            except Exception as exc:
                prediction = {
                    "status": "failed",
                    "text": None,
                    "rouge": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            example["predictions"][model_key] = prediction
        run["models"][model_key].update(_model_summary(run["examples"], model_key))
        save_run("evaluation_a", run)
        del generator
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except (ImportError, RuntimeError):
            pass
    run["status"] = "completed"
    run["completed_at"] = datetime.now(UTC).isoformat()
    save_run("evaluation_a", run)
    return run
