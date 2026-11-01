"""Sequential PostgreSQL-backed generation worker."""

import gc
import json
import time
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from uuid import UUID

from sqlalchemy import Engine, select, text, update

from related_work_assistant.core.config import Settings
from related_work_assistant.database.models import (
    ModelVersion,
    RelatedWorkGeneration,
    RelatedWorkJob,
)
from related_work_assistant.database.session import session_scope
from related_work_assistant.generation.evidence import build_evidence_snapshot, evidence_hash
from related_work_assistant.generation.generators import (
    PrimeraGenerator,
    QwenGenerator,
    QwenLoraGenerator,
)
from related_work_assistant.generation.service import model_config


def recover_stale_jobs(engine: Engine, minutes: int) -> int:
    cutoff = datetime.now(UTC) - timedelta(minutes=minutes)
    with session_scope(engine) as session:
        result = session.execute(
            update(RelatedWorkJob)
            .where(RelatedWorkJob.status == "running", RelatedWorkJob.started_at < cutoff)
            .values(
                status="pending",
                started_at=None,
                error_message="Recovered after worker interruption",
            )
        )
        return result.rowcount


def claim_job(engine: Engine) -> UUID | None:
    with session_scope(engine) as session:
        job = session.scalar(
            select(RelatedWorkJob)
            .where(RelatedWorkJob.status == "pending")
            .order_by(RelatedWorkJob.created_at, RelatedWorkJob.id)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if job is None:
            return None
        job.status = "running"
        job.started_at = datetime.now(UTC)
        return job.id


def _register_model(engine: Engine, generator, generation_version: str) -> UUID:
    identity = {
        "repository_id": generator.repository_id,
        "revision": generator.revision,
        "generation_version": generation_version,
        "adapter_revision": getattr(generator, "adapter_revision", None),
    }
    key = (
        "generation:"
        + generator.model_key
        + ":"
        + sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    )
    with session_scope(engine) as session:
        row = session.scalar(select(ModelVersion).where(ModelVersion.model_key == key))
        if row is None:
            row = ModelVersion(
                model_key=key,
                role="generation",
                provider="huggingface",
                repository_id=generator.repository_id,
                checkpoint_revision=generator.revision,
                adapter_repository_id=getattr(generator, "adapter_repository_id", None),
                adapter_revision=getattr(generator, "adapter_revision", None),
                parameter_count=generator.parameter_count,
                config={
                    "library": "peft"
                    if getattr(generator, "adapter_revision", None)
                    else "transformers",
                    "generation_version": generation_version,
                },
            )
            session.add(row)
            session.flush()
        return row.id


def process_job(engine: Engine, settings: Settings, job_id: UUID) -> None:
    with session_scope(engine) as session:
        job = session.get(RelatedWorkJob, job_id)
        if job is None or job.status != "running":
            return
        paper_id, model_key, force, cache_key = (
            job.paper_id,
            job.model_key,
            job.force_regenerate,
            job.cache_key,
        )
    config = model_config(settings, model_key)
    snapshot: dict[str, object] = {}
    snapshot_hash = sha256(b"{}").hexdigest()
    generator = None
    try:
        snapshot = build_evidence_snapshot(engine, paper_id, model_key)
        snapshot_hash = evidence_hash(snapshot)
        common = dict(
            cache_dir=settings.model_cache_dir,
            device=settings.generation_device,
            max_new_tokens=int(config["max_new_tokens"]),
            safety_margin=int(config["safety_margin"]),
        )
        if model_key == "primera":
            generator = PrimeraGenerator(
                revision=settings.primera_revision,
                min_new_tokens=int(config["min_new_tokens"]),
                num_beams=int(config["num_beams"]),
                length_penalty=float(config["length_penalty"]),
                **common,
            )
        elif model_key == "qwen":
            generator = QwenGenerator(revision=settings.qwen_revision, **common)
        elif model_key == "qwen_lora":
            if not config["adapter_available"] or not config["adapter_revision"]:
                raise RuntimeError("qwen_lora adapter is unavailable or invalid")
            generator = QwenLoraGenerator(
                revision=str(config["base_model_revision"]),
                adapter_path=settings.qwen_lora_dir / "adapter",
                adapter_revision=str(config["adapter_revision"]),
                **common,
            )
        else:
            raise ValueError(f"Unsupported generation model: {model_key}")
        if not force:
            with session_scope(engine) as session:
                cached = session.scalar(
                    select(RelatedWorkGeneration)
                    .where(
                        RelatedWorkGeneration.paper_id == paper_id,
                        RelatedWorkGeneration.model_key == model_key,
                        RelatedWorkGeneration.evidence_hash == snapshot_hash,
                        RelatedWorkGeneration.cache_key == cache_key,
                        RelatedWorkGeneration.model_revision == generator.revision,
                        RelatedWorkGeneration.status == "completed",
                    )
                    .order_by(RelatedWorkGeneration.created_at.desc())
                    .limit(1)
                )
                if cached:
                    job = session.get(RelatedWorkJob, job_id)
                    job.status, job.generation_id, job.finished_at = (
                        "completed",
                        cached.id,
                        datetime.now(UTC),
                    )
                    job.error_message = None
                    return
        peak_memory_mb = None
        generator_device = getattr(generator, "device", "cpu")
        if generator_device == "cuda":
            import torch

            torch.cuda.reset_peak_memory_stats()
        inference_started = time.perf_counter()
        output = generator.generate(snapshot)
        inference_latency_ms = (time.perf_counter() - inference_started) * 1000
        if generator_device == "cuda":
            import torch

            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        model_version_id = (
            _register_model(engine, generator, str(config["generation_version"]))
            if output.status in ("completed", "invalid")
            else None
        )
        references = snapshot.get("chunk_pairs", snapshot.get("paper_retrieval", []))
        with session_scope(engine) as session:
            generation = RelatedWorkGeneration(
                paper_id=paper_id,
                model_key=model_key,
                model_version_id=model_version_id,
                model_repository_id=generator.repository_id,
                model_revision=generator.revision,
                prompt_version=str(config["prompt_version"]),
                generation_version=str(config["generation_version"]),
                generation_config=config,
                cache_key=cache_key,
                evidence_hash=snapshot_hash,
                input_hash=output.input_hash,
                evidence_snapshot=snapshot,
                generated_text=output.text,
                citations=output.citations,
                source_references=references,
                input_token_count=output.input_tokens,
                output_token_count=output.output_tokens,
                context_limit=output.context_limit,
                inference_latency_ms=inference_latency_ms,
                peak_memory_mb=peak_memory_mb,
                status=output.status,
                error_message=output.error,
                finish_reason=output.finish_reason,
                validation_errors=list(output.validation_errors),
                completed_at=datetime.now(UTC),
            )
            session.add(generation)
            session.flush()
            job = session.get(RelatedWorkJob, job_id)
            job.status, job.generation_id, job.finished_at = (
                "completed",
                generation.id,
                datetime.now(UTC),
            )
            job.error_message = None
    except Exception as exc:
        detail = str(exc).strip()
        message = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
        with session_scope(engine) as session:
            generation = RelatedWorkGeneration(
                paper_id=paper_id,
                model_key=model_key,
                model_repository_id=str(config["repository_id"]),
                model_revision=(
                    generator.revision
                    if generator is not None
                    else str(config["requested_revision"] or "unresolved")
                ),
                prompt_version=str(config["prompt_version"]),
                generation_version=str(config["generation_version"]),
                generation_config=config,
                cache_key=cache_key,
                evidence_hash=snapshot_hash,
                input_hash=sha256(b"").hexdigest(),
                evidence_snapshot=snapshot,
                generated_text=None,
                citations=[],
                source_references=snapshot.get("chunk_pairs", snapshot.get("paper_retrieval", [])),
                status="failed",
                error_message=message,
                validation_errors=[message],
                completed_at=datetime.now(UTC),
            )
            session.add(generation)
            session.flush()
            job = session.get(RelatedWorkJob, job_id)
            if job:
                job.status = "failed"
                job.generation_id = generation.id
                job.error_message = message
                job.finished_at = datetime.now(UTC)
    finally:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass


def run_worker(engine: Engine, settings: Settings, *, once: bool = False) -> None:
    with engine.connect() as worker_lock:
        acquired = worker_lock.scalar(
            text("SELECT pg_try_advisory_lock(hashtext('related-work-generation-worker'))")
        )
        if acquired is not True:
            raise RuntimeError(
                "Another generation worker is already active; use that single process"
            )
        try:
            recover_stale_jobs(engine, settings.generation_stale_minutes)
            while True:
                job_id = claim_job(engine)
                if job_id:
                    process_job(engine, settings, job_id)
                elif once:
                    return
                else:
                    time.sleep(settings.generation_poll_seconds)
                if once:
                    return
        finally:
            worker_lock.execute(
                text("SELECT pg_advisory_unlock(hashtext('related-work-generation-worker'))")
            )
