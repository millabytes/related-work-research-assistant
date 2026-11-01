"""Lightweight Qwen LoRA artifact discovery shared by CLI and API processes."""

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from related_work_assistant.core.config import PROJECT_ROOT

QWEN_LORA_ROOT = PROJECT_ROOT / "models_cache/qwen_lora"
QWEN_LORA_SMOKE_ROOT = PROJECT_ROOT / "models_cache/qwen_lora-smoke"
QWEN_BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def adapter_directory(root: Path = QWEN_LORA_ROOT) -> Path:
    return root / "adapter"


def adapter_fingerprint(path: Path) -> str:
    """Hash the immutable adapter config and weights, independent of filesystem metadata."""
    digest = sha256()
    files = [path / "adapter_config.json"]
    weights = path / "adapter_model.safetensors"
    files.append(weights if weights.is_file() else path / "adapter_model.bin")
    for file in files:
        if not file.is_file():
            raise FileNotFoundError(f"Missing LoRA adapter artifact: {file}")
        digest.update(file.name.encode())
        with file.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def adapter_status(root: Path = QWEN_LORA_ROOT) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    path = adapter_directory(root)
    if not manifest_path.is_file():
        return {
            "available": False,
            "status": "not_available",
            "reason": f"Missing training manifest: {manifest_path}",
            "root": str(root),
            "adapter_path": str(path),
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fingerprint = adapter_fingerprint(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "status": "invalid_artifact",
            "reason": str(exc),
            "root": str(root),
            "adapter_path": str(path),
        }
    expected = manifest.get("adapter_revision")
    base_revision = str(manifest.get("base_model_revision") or "")
    if (
        manifest.get("status") != "completed"
        or expected != fingerprint
        or manifest.get("base_model_id") != QWEN_BASE_MODEL_ID
        or len(base_revision) != 40
    ):
        return {
            "available": False,
            "status": "invalid_artifact",
            "reason": "Manifest is incomplete or adapter fingerprint does not match",
            "root": str(root),
            "adapter_path": str(path),
            "manifest": manifest,
        }
    return {
        "available": True,
        "status": "available",
        "reason": None,
        "root": str(root),
        "adapter_path": str(path),
        "adapter_revision": fingerprint,
        "manifest": manifest,
    }


def training_artifact_status(root: Path = QWEN_LORA_ROOT) -> dict[str, Any]:
    status_path = root / "status.json"
    metrics_path = root / "metrics.json"
    checkpoints_dir = root / "checkpoints"
    checkpoints = []
    if checkpoints_dir.is_dir():
        checkpoints = sorted(
            (path for path in checkpoints_dir.glob("checkpoint-*") if path.is_dir()),
            key=lambda path: int(path.name.rsplit("-", 1)[-1]),
        )
    result = adapter_status(root)
    result["training_status"] = (
        json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else None
    )
    result["metrics"] = (
        json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else None
    )
    result["checkpoints"] = [str(path) for path in checkpoints]
    return result
