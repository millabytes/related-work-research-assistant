"""Evaluation B: citation and NLI grounding over persisted application generations."""

import json
import random
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Engine, select

from related_work_assistant.core.config import Settings
from related_work_assistant.database.models import Paper, RelatedWorkGeneration
from related_work_assistant.database.session import session_scope
from related_work_assistant.evaluation.generation.metrics import (
    aggregate_numeric,
    citation_metrics,
    link_claims_to_evidence,
)
from related_work_assistant.evaluation.generation.nli import LocalNliScorer, score_grounding_links
from related_work_assistant.evaluation.generation.storage import (
    EVALUATION_ROOT,
    atomic_json,
    new_run_id,
    save_run,
)

MODEL_KEYS = ("primera", "qwen", "qwen_lora")
FIXED_SET_PATH = EVALUATION_ROOT / "evaluation_b/fixed_papers.json"


def _latest_generations(engine: Engine, paper_id: UUID) -> dict[str, RelatedWorkGeneration | None]:
    with session_scope(engine) as session:
        return {
            model_key: session.scalar(
                select(RelatedWorkGeneration)
                .where(
                    RelatedWorkGeneration.paper_id == paper_id,
                    RelatedWorkGeneration.model_key == model_key,
                )
                .order_by(
                    RelatedWorkGeneration.created_at.desc(), RelatedWorkGeneration.id.desc()
                )
                .limit(1)
            )
            for model_key in MODEL_KEYS
        }


def _same_qwen_input(
    qwen: RelatedWorkGeneration | None, qwen_lora: RelatedWorkGeneration | None
) -> bool | None:
    if qwen is None or qwen_lora is None:
        return None
    first = dict(qwen.evidence_snapshot)
    second = dict(qwen_lora.evidence_snapshot)
    first.pop("model_key", None)
    second.pop("model_key", None)
    return first == second and qwen.prompt_version == qwen_lora.prompt_version


def _evaluate_row(
    row: RelatedWorkGeneration | None,
    *,
    model_key: str,
    scorer: Any | None,
    nli_error: str | None,
) -> dict[str, Any]:
    if row is None:
        return {
            "model_key": model_key,
            "status": "not_evaluated",
            "generation_status": "missing",
            "metrics": None,
            "claims": [],
        }
    base = {
        "model_key": model_key,
        "generation_id": str(row.id),
        "generation_status": row.status,
        "generated_text": row.generated_text,
        "error_message": row.error_message,
        "model_revision": row.model_revision,
        "adapter_revision": row.generation_config.get("adapter_revision"),
        "adapter_fingerprint": row.generation_config.get("adapter_revision"),
        "prompt_version": row.prompt_version,
        "evidence_hash": row.evidence_hash,
        "inference_latency_ms": row.inference_latency_ms,
        "peak_memory_mb": row.peak_memory_mb,
        "included_in_aggregate": row.status == "completed",
    }
    if not row.generated_text:
        return {**base, "status": "no_output", "metrics": None, "claims": []}
    if model_key == "primera":
        return {
            **base,
            "status": "evaluated",
            "citation_metrics": citation_metrics(row.generated_text, [], unresolved=True),
            "nli": {
                "status": "unresolved",
                "reason": "PRIMERA does not emit reliably source-specific citation identifiers.",
                "entailment": None,
                "neutral": None,
                "contradiction": None,
            },
            "claims": [],
        }
    allowed = [
        str(paper["citation_key"])
        for paper in row.evidence_snapshot.get("recommended_papers", [])
    ]
    citations = citation_metrics(row.generated_text, allowed)
    links = link_claims_to_evidence(row.generated_text, row.evidence_snapshot)
    if scorer is not None:
        claims, nli = score_grounding_links(links, scorer)
        nli = {"status": "evaluated" if nli["evaluated_claim_sources"] else "not_evaluated", **nli}
    else:
        claims = [{**link, "nli": None, "evidence_scores": []} for link in links]
        nli = {
            "status": "failed" if nli_error else "not_evaluated",
            "reason": nli_error or "No source-linked claims were available.",
            "evaluated_claim_sources": 0,
            "entailment": None,
            "neutral": None,
            "contradiction": None,
        }
    return {
        **base,
        "status": "evaluated",
        "citation_metrics": citations,
        "nli": nli,
        "claims": claims,
    }


def _aggregate(papers: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model_key in MODEL_KEYS:
        models = [paper["models"][model_key] for paper in papers]
        completed = [model for model in models if model.get("included_in_aggregate")]
        flat = []
        for model in completed:
            citations = model.get("citation_metrics") or {}
            nli = model.get("nli") or {}
            flat.append(
                {
                    "citation_validity": citations.get("validity"),
                    "citation_coverage": citations.get("coverage"),
                    "nli_entailment": nli.get("entailment"),
                    "nli_neutral": nli.get("neutral"),
                    "nli_contradiction": nli.get("contradiction"),
                    "inference_latency_ms": model.get("inference_latency_ms"),
                    "peak_memory_mb": model.get("peak_memory_mb"),
                }
            )
        result[model_key] = {
            "available_outputs": len(completed),
            "missing_or_rejected": len(models) - len(completed),
            **aggregate_numeric(
                flat,
                (
                    "citation_validity",
                    "citation_coverage",
                    "nli_entailment",
                    "nli_neutral",
                    "nli_contradiction",
                    "inference_latency_ms",
                    "peak_memory_mb",
                ),
            ),
        }
    return result


def run_evaluation_b(
    engine: Engine,
    settings: Settings,
    paper_ids: list[UUID],
    *,
    device: str = "auto",
    nli_revision: str | None = None,
    fixed_set: bool = False,
) -> dict[str, Any]:
    run = {
        "run_id": new_run_id("eval-b"),
        "kind": "evaluation_b",
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "config": {
            "paper_ids": [str(value) for value in paper_ids],
            "device": device,
            "nli_model": "cross-encoder/nli-MiniLM2-L6-H768",
            "nli_is_proxy_not_scientific_truth": True,
            "nli_evidence_selection": (
                "maximum entailment among persisted recommended chunks of the cited paper only"
            ),
            "fixed_set": fixed_set,
        },
        "sample_ids": [str(value) for value in paper_ids],
        "papers": [],
    }
    save_run("evaluation_b", run)
    rows_by_paper = {paper_id: _latest_generations(engine, paper_id) for paper_id in paper_ids}
    needs_nli = any(
        row is not None and row.generated_text
        for rows in rows_by_paper.values()
        for key, row in rows.items()
        if key in ("qwen", "qwen_lora")
    )
    scorer = None
    nli_error = None
    if needs_nli:
        try:
            scorer = LocalNliScorer(
                cache_dir=settings.model_cache_dir,
                device=device,
                revision=nli_revision,
            )
            run["config"]["nli_revision"] = scorer.revision
        except Exception as exc:
            nli_error = f"{type(exc).__name__}: {exc}"
            run["config"]["nli_error"] = nli_error
    with session_scope(engine) as session:
        titles = {
            paper_id: session.scalar(select(Paper.title).where(Paper.id == paper_id))
            for paper_id in paper_ids
        }
    for paper_id in paper_ids:
        rows = rows_by_paper[paper_id]
        paper = {
            "paper_id": str(paper_id),
            "title": titles.get(paper_id),
            "qwen_vs_qwen_lora_identical_input": _same_qwen_input(
                rows["qwen"], rows["qwen_lora"]
            ),
            "models": {
                key: _evaluate_row(rows[key], model_key=key, scorer=scorer, nli_error=nli_error)
                for key in MODEL_KEYS
            },
        }
        run["papers"].append(paper)
        save_run("evaluation_b", run)
    run["aggregate"] = _aggregate(run["papers"])
    run["comparison_notes"] = {
        "qwen_vs_qwen_lora": (
            "Direct model comparison only where persisted prompt versions and evidence snapshots match."
        ),
        "primera": (
            "PRIMERA is an abstract-only complete-system baseline; its comparison with Qwen "
            "is not an isolated same-input model comparison."
        ),
    }
    run["status"] = "completed"
    run["completed_at"] = datetime.now(UTC).isoformat()
    save_run("evaluation_b", run)
    return run


def fixed_paper_ids(
    engine: Engine, *, limit: int = 10, seed: int = 42, reset: bool = False
) -> list[UUID]:
    if FIXED_SET_PATH.is_file() and not reset:
        payload = json.loads(FIXED_SET_PATH.read_text(encoding="utf-8"))
        return [UUID(value) for value in payload["paper_ids"]]
    with session_scope(engine) as session:
        candidates = list(
            session.scalars(
                select(RelatedWorkGeneration.paper_id)
                .distinct()
                .order_by(RelatedWorkGeneration.paper_id)
            )
        )
    selected = (
        random.Random(seed).sample(candidates, limit) if len(candidates) > limit else candidates
    )
    atomic_json(
        FIXED_SET_PATH,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "seed": seed,
            "requested_size": limit,
            "paper_ids": [str(value) for value in selected],
        },
    )
    return selected
