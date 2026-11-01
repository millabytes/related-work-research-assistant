"""Queueing, caching and status reads for related-work generation."""

import json
from hashlib import sha256
from uuid import UUID

from sqlalchemy import Engine, exists, func, select, text
from sqlalchemy.dialects.postgresql import insert

from related_work_assistant.core.config import Settings
from related_work_assistant.database.models import (
    Chunk,
    Paper,
    RelatedWorkGeneration,
    RelatedWorkJob,
)
from related_work_assistant.database.session import session_scope
from related_work_assistant.generation.generators import (
    PRIMERA_GENERATION_VERSION,
    PRIMERA_PROMPT_VERSION,
    PRIMERA_REPOSITORY,
    QWEN_GENERATION_VERSION,
    QWEN_LORA_GENERATION_VERSION,
    QWEN_PROMPT_VERSION,
    QWEN_REPOSITORY,
    build_primera_input,
    build_qwen_messages,
)
from related_work_assistant.retrieval.papers import recommend_papers
from related_work_assistant.training.lora_artifacts import adapter_status

DEFAULT_MODELS = ("primera", "qwen")
AVAILABLE_MODELS = (*DEFAULT_MODELS, "qwen_lora")
READABLE_MODELS = (*AVAILABLE_MODELS, "gemma")


def model_config(settings: Settings, model_key: str) -> dict[str, object]:
    if model_key == "primera":
        return {
            "repository_id": PRIMERA_REPOSITORY,
            "requested_revision": settings.primera_revision,
            "max_new_tokens": settings.primera_max_new_tokens,
            "min_new_tokens": settings.primera_min_new_tokens,
            "num_beams": settings.primera_num_beams,
            "length_penalty": settings.primera_length_penalty,
            "no_repeat_ngram_size": 3,
            "global_attention": "bos_and_doc_separators",
            "source_format": "target_abstract_then_reference_abstracts",
            "safety_margin": 0,
            "prompt_version": PRIMERA_PROMPT_VERSION,
            "generation_version": PRIMERA_GENERATION_VERSION,
        }
    if model_key in ("qwen", "qwen_lora"):
        config: dict[str, object] = {
            "repository_id": QWEN_REPOSITORY,
            "requested_revision": settings.qwen_revision,
            "max_new_tokens": settings.qwen_max_new_tokens,
            "safety_margin": settings.generation_safety_margin,
            "do_sample": False,
            "truncation": False,
            "prompt_version": QWEN_PROMPT_VERSION,
            "generation_version": QWEN_GENERATION_VERSION,
            "reranker_revision": settings.reranker_revision,
            "reranker_max_length": settings.reranker_max_length,
        }
        if model_key == "qwen_lora":
            status = adapter_status(settings.qwen_lora_dir)
            manifest = status.get("manifest") or {}
            config.update(
                {
                    "adapter_path": status["adapter_path"],
                    "adapter_available": status["available"],
                    "adapter_revision": status.get("adapter_revision"),
                    "base_model_revision": manifest.get("base_model_revision"),
                    "requested_revision": manifest.get("base_model_revision"),
                    "generation_version": QWEN_LORA_GENERATION_VERSION,
                }
            )
        return config
    raise ValueError(f"Unsupported model: {model_key}")


def evidence_state_hash(engine: Engine, paper_id: UUID, model_key: str) -> str:
    recommendations = recommend_papers(engine, paper_id, top_k=5)
    paper_ids = [paper_id, *(item.paper_id for item in recommendations)]
    with session_scope(engine) as session:
        papers = list(
            session.execute(
                select(
                    Paper.id,
                    Paper.title,
                    Paper.abstract,
                    Paper.publication_year,
                    Paper.venue_name,
                    Paper.citation_key,
                    Paper.updated_at,
                )
                .where(Paper.id.in_(paper_ids))
                .order_by(Paper.id)
            )
        )
        chunks = []
        if model_key in ("qwen", "qwen_lora"):
            chunks = list(
                session.execute(
                    select(Chunk.id, Chunk.paper_id, Chunk.content_sha256, Chunk.section_id)
                    .where(Chunk.paper_id.in_(paper_ids))
                    .order_by(Chunk.id)
                )
            )
    return sha256(
        json.dumps(
            {
                "recommendations": [(str(item.paper_id), item.rank) for item in recommendations],
                "papers": [tuple(str(value) for value in row) for row in papers],
                "chunks": [tuple(str(value) for value in row) for row in chunks],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def request_cache_key(engine: Engine, settings: Settings, paper_id: UUID, model_key: str) -> str:
    payload = {
        "paper_id": str(paper_id),
        "model_key": model_key,
        "evidence_state_hash": evidence_state_hash(engine, paper_id, model_key),
        **model_config(settings, model_key),
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def enqueue(
    engine: Engine,
    settings: Settings,
    paper_id: UUID,
    model_keys: list[str] | None = None,
    *,
    force: bool = False,
) -> list[UUID]:
    selected = list(dict.fromkeys(model_keys or DEFAULT_MODELS))
    if not selected or any(key not in AVAILABLE_MODELS for key in selected):
        raise ValueError("models must contain primera, qwen, and/or qwen_lora")
    job_ids: list[UUID] = []
    with session_scope(engine) as session:
        if session.scalar(select(exists().where(Paper.id == paper_id))) is not True:
            raise LookupError(f"Paper {paper_id} was not found")
        for model_key in selected:
            config = model_config(settings, model_key)
            if model_key == "qwen_lora" and not config["adapter_available"]:
                raise ValueError(
                    "qwen_lora is unavailable; complete training and create a valid adapter first"
                )
            cache_key = request_cache_key(engine, settings, paper_id, model_key)
            revision_is_pinned = bool(
                config["requested_revision"] or config.get("adapter_revision")
            )
            if not force and revision_is_pinned:
                cached = session.scalar(
                    select(RelatedWorkGeneration.id)
                    .where(
                        RelatedWorkGeneration.paper_id == paper_id,
                        RelatedWorkGeneration.model_key == model_key,
                        RelatedWorkGeneration.cache_key == cache_key,
                        RelatedWorkGeneration.status == "completed",
                    )
                    .limit(1)
                )
                if cached:
                    continue
            statement = (
                insert(RelatedWorkJob)
                .values(
                    paper_id=paper_id,
                    model_key=model_key,
                    force_regenerate=force,
                    cache_key=cache_key,
                )
                .on_conflict_do_nothing(
                    index_elements=["paper_id", "model_key"],
                    index_where=text("status IN ('pending','running')"),
                )
                .returning(RelatedWorkJob.id)
            )
            job_id = session.scalar(statement)
            if job_id:
                job_ids.append(job_id)
    return job_ids


def _bulk_regenerate_model(
    engine: Engine, settings: Settings, model_key: str
) -> dict[str, object]:
    """Force-enqueue one model job per paper while preserving active-job uniqueness."""
    with session_scope(engine) as session:
        paper_ids = list(session.scalars(select(Paper.id).order_by(Paper.id)))
    queued = 0
    already_active = 0
    failures: list[dict[str, str]] = []
    for paper_id in paper_ids:
        try:
            job_ids = enqueue(engine, settings, paper_id, [model_key], force=True)
        except Exception as exc:
            failures.append(
                {
                    "paper_id": str(paper_id),
                    "error": str(exc)
                    if isinstance(exc, (LookupError, RuntimeError, ValueError))
                    else type(exc).__name__,
                }
            )
            continue
        if job_ids:
            queued += 1
        else:
            already_active += 1
    return {
        "papers": len(paper_ids),
        "queued": queued,
        "already_active": already_active,
        "failed_to_queue": len(failures),
        "failures": failures,
    }


def bulk_regenerate_qwen(engine: Engine, settings: Settings) -> dict[str, object]:
    return _bulk_regenerate_model(engine, settings, "qwen")


def bulk_regenerate_qwen_lora(engine: Engine, settings: Settings) -> dict[str, object]:
    return _bulk_regenerate_model(engine, settings, "qwen_lora")


def _generation_job_status(engine: Engine, model_key: str) -> dict[str, object]:
    """Return aggregate job and current-prompt result counts for CLI monitoring."""
    generation_version = (
        QWEN_LORA_GENERATION_VERSION if model_key == "qwen_lora" else QWEN_GENERATION_VERSION
    )
    with session_scope(engine) as session:
        raw_jobs = dict(
            session.execute(
                select(RelatedWorkJob.status, func.count(RelatedWorkJob.id))
                .where(RelatedWorkJob.model_key == model_key)
                .group_by(RelatedWorkJob.status)
            ).all()
        )
        result_counts = dict(
            session.execute(
                select(RelatedWorkGeneration.status, func.count(RelatedWorkGeneration.id))
                .where(
                    RelatedWorkGeneration.model_key == model_key,
                    RelatedWorkGeneration.prompt_version == QWEN_PROMPT_VERSION,
                    RelatedWorkGeneration.generation_version == generation_version,
                )
                .group_by(RelatedWorkGeneration.status)
            ).all()
        )
        total_papers = session.scalar(select(func.count(Paper.id))) or 0
    return {
        "model": model_key,
        "prompt_version": QWEN_PROMPT_VERSION,
        "papers": total_papers,
        "jobs": {
            "queued": raw_jobs.get("pending", 0),
            "running": raw_jobs.get("running", 0),
            "completed": raw_jobs.get("completed", 0),
            "failed": raw_jobs.get("failed", 0),
        },
        "current_prompt_results": result_counts,
    }


def qwen_job_status(engine: Engine) -> dict[str, object]:
    return _generation_job_status(engine, "qwen")


def qwen_lora_job_status(engine: Engine) -> dict[str, object]:
    return _generation_job_status(engine, "qwen_lora")


def _generation_dict(
    row: RelatedWorkGeneration, *, include_model_input: bool = False
) -> dict[str, object]:
    payload: dict[str, object] = {
        "generation_id": row.id,
        "model_key": row.model_key,
        "model_repository_id": row.model_repository_id,
        "model_revision": row.model_revision,
        "prompt_version": row.prompt_version,
        "generation_version": row.generation_version,
        "generation_config": row.generation_config,
        "generated_text": row.generated_text,
        "citations": row.citations,
        "source_references": row.source_references,
        "input_token_count": row.input_token_count,
        "output_token_count": row.output_token_count,
        "context_limit": row.context_limit,
        "inference_latency_ms": row.inference_latency_ms,
        "peak_memory_mb": row.peak_memory_mb,
        "status": row.status,
        "error_message": row.error_message,
        "finish_reason": row.finish_reason,
        "validation_errors": row.validation_errors,
        "evidence_hash": row.evidence_hash,
        "input_hash": row.input_hash,
        "created_at": row.created_at,
        "completed_at": row.completed_at,
    }
    if include_model_input:
        rendered_input: dict[str, object] | None = None
        try:
            if row.model_key in ("qwen", "qwen_lora") and row.prompt_version == QWEN_PROMPT_VERSION:
                rendered_input = {"messages": build_qwen_messages(row.evidence_snapshot)}
            elif row.model_key == "primera" and row.prompt_version == PRIMERA_PROMPT_VERSION:
                rendered_input = {"source": build_primera_input(row.evidence_snapshot)}
        except (KeyError, TypeError, ValueError):
            # Persisted evidence remains inspectable even if an old or partial
            # snapshot cannot be rendered with the current prompt builder.
            rendered_input = None
        payload.update(
            {
                "evidence_snapshot": row.evidence_snapshot,
                "rendered_model_input": rendered_input,
            }
        )
    return payload


def related_work_status(
    engine: Engine, paper_id: UUID, model_key: str | None = None
) -> dict[str, object]:
    keys = [model_key] if model_key else list(AVAILABLE_MODELS)
    if any(key not in READABLE_MODELS for key in keys):
        raise ValueError("model must be primera, qwen, qwen_lora, or historical gemma")
    with session_scope(engine) as session:
        if session.scalar(select(exists().where(Paper.id == paper_id))) is not True:
            raise LookupError(f"Paper {paper_id} was not found")
        models = []
        for key in keys:
            availability = (
                adapter_status(Settings().qwen_lora_dir)
                if key == "qwen_lora"
                else {"available": True, "status": "available", "reason": None}
            )
            history = list(
                session.scalars(
                    select(RelatedWorkGeneration)
                    .where(
                        RelatedWorkGeneration.paper_id == paper_id,
                        RelatedWorkGeneration.model_key == key,
                    )
                    .order_by(
                        RelatedWorkGeneration.created_at.desc(), RelatedWorkGeneration.id.desc()
                    )
                )
            )
            latest_job = session.scalar(
                select(RelatedWorkJob)
                .where(RelatedWorkJob.paper_id == paper_id, RelatedWorkJob.model_key == key)
                .order_by(RelatedWorkJob.created_at.desc())
                .limit(1)
            )
            job = latest_job if latest_job and latest_job.status in ("pending", "running") else None
            active = (
                None
                if job is None
                else {
                    "job_id": job.id,
                    "status": job.status,
                    "created_at": job.created_at,
                    "started_at": job.started_at,
                }
            )
            models.append(
                {
                    "model_key": key,
                    "available": availability["available"],
                    "unavailable_reason": availability.get("reason"),
                    "adapter_revision": availability.get("adapter_revision"),
                    "status": "not_available"
                    if not availability["available"] and not history and not job
                    else job.status
                    if job
                    else (
                        history[0].status
                        if history
                        else (latest_job.status if latest_job else "missing")
                    ),
                    "last_job_error": latest_job.error_message if latest_job else None,
                    "active_job": active,
                    "latest": (
                        _generation_dict(history[0], include_model_input=True) if history else None
                    ),
                    "history": [_generation_dict(item) for item in history],
                }
            )
        return {"paper_id": paper_id, "models": models}
