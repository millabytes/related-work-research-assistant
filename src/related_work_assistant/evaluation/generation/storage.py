"""Atomic, append-only evaluation result storage with replaceable latest pointers."""

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from related_work_assistant.core.config import PROJECT_ROOT

EVALUATION_ROOT = PROJECT_ROOT / "data/evaluations/generation"


def new_run_id(prefix: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{timestamp}-{uuid4().hex[:8]}"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
        json.dump(value, file, ensure_ascii=False, indent=2, default=str)
        file.write("\n")
        temporary = Path(file.name)
    os.replace(temporary, path)


def save_run(kind: str, payload: dict[str, Any]) -> Path:
    directory = EVALUATION_ROOT / kind
    path = directory / f"{payload['run_id']}.json"
    atomic_json(path, payload)
    atomic_json(directory / "latest.json", payload)
    return path


def load_latest(kind: str) -> dict[str, Any] | None:
    path = EVALUATION_ROOT / kind / "latest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def list_runs(kind: str, *, limit: int = 20) -> list[dict[str, Any]]:
    directory = EVALUATION_ROOT / kind
    paths = sorted(
        (path for path in directory.glob("*.json") if path.name != "latest.json"),
        reverse=True,
    )[:limit]
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def latest_evidence_for_paper(paper_id: str) -> dict[str, Any] | None:
    for run in list_runs("evaluation_b", limit=100):
        for paper in run.get("papers", []):
            if str(paper.get("paper_id")) == paper_id:
                return {"run": {k: v for k, v in run.items() if k != "papers"}, "paper": paper}
    return None
