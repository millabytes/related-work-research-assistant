import typer

from related_work_assistant.core.config import Settings
from related_work_assistant.core.logging import configure_logging
from related_work_assistant.database.health import check_database
from related_work_assistant.database.session import make_engine

app = typer.Typer(help="Related Work Assistant infrastructure.", pretty_exceptions_enable=False)


def _paper_ids(values: list[str] | None):
    from uuid import UUID

    try:
        return [UUID(value) for value in values] if values else None
    except ValueError as exc:
        raise typer.BadParameter("--paper-id must be a UUID") from exc


@app.callback()
def main() -> None:
    """Local research assistant commands."""


@app.command()
def db_check() -> None:
    """Check PostgreSQL connectivity and the vector extension."""
    engine = None
    try:
        settings = Settings()
        configure_logging(settings.log_level)
        engine = make_engine(settings)
        if not check_database(engine):
            raise RuntimeError("Missing vector extension")
    except Exception:
        typer.echo(
            "Database unavailable or vector extension missing; check configuration and migrations.",
            err=True,
        )
        raise typer.Exit(1) from None
    finally:
        if engine is not None:
            engine.dispose()
    typer.echo("Database ready (PostgreSQL + vector).")


@app.command()
def ingest(
    path: str = typer.Argument(..., help="PDF file or directory, relative to STORAGE_ROOT."),
    limit: int | None = typer.Option(None, min=1, help="Maximum PDFs to process."),
    force: bool = typer.Option(
        False, help="Re-extract; preserve existing paper metadata for review."
    ),
) -> None:
    """Extract PDF text with PyMuPDF, metadata with local GROBID, and persist results."""
    from pathlib import Path

    from related_work_assistant.extraction.grobid import GrobidClient
    from related_work_assistant.pipelines.ingestion import ingest_pdf, write_json

    settings = Settings()
    target = (settings.storage_root / Path(path)).resolve()
    if not target.exists() or not target.is_relative_to(settings.storage_root):
        raise typer.BadParameter("Path must exist inside STORAGE_ROOT")
    files = (
        sorted(p for p in target.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")
        if target.is_dir()
        else [target]
    )
    if limit:
        files = files[:limit]
    if not files:
        raise typer.BadParameter("No PDFs found")
    engine = make_engine(settings)
    results = []
    try:
        with GrobidClient(settings.grobid_url, settings.grobid_timeout) as client:
            for pdf in files:
                try:
                    result = ingest_pdf(pdf, settings, engine, client, force=force)
                except Exception as exc:
                    result = {
                        "path": str(pdf.relative_to(settings.storage_root)),
                        "status": "failed",
                        "error": str(exc)
                        if isinstance(exc, (ValueError, RuntimeError))
                        else type(exc).__name__,
                    }
                results.append(result)
                typer.echo(f"{result['status']}: {pdf.name}")
                write_json(settings.processed_dir / "ingestion_manifest.json", results)
    finally:
        engine.dispose()
    typer.echo(f"Manifest: {settings.processed_dir / 'ingestion_manifest.json'}")
    if any(r["status"] == "failed" for r in results):
        raise typer.Exit(1)


@app.command()
def evaluate_metadata() -> None:
    """Compare the latest ingestion manifest to Zotero RDF, with per-field metrics."""
    from related_work_assistant.evaluation.report import evaluate_manifest

    settings = Settings()
    output = settings.storage_root / "data/evaluations/metadata_evaluation.json"
    report = evaluate_manifest(
        settings.processed_dir / "ingestion_manifest.json",
        settings.zotero_rdf,
        settings.storage_root,
        settings.raw_pdfs_dir,
        output,
    )
    typer.echo(f"Evaluated {report['evaluated_unique_pdfs']} unique PDFs. Report: {output}")


@app.command()
def show_metadata(paper_id: str) -> None:
    """Read a paper's structured metadata back from PostgreSQL as JSON."""
    from uuid import UUID

    from related_work_assistant.pipelines.ingestion import load_metadata

    engine = make_engine(Settings())
    try:
        typer.echo(load_metadata(engine, UUID(paper_id)).model_dump_json(indent=2))
    finally:
        engine.dispose()


@app.command("generation-worker")
def generation_worker(
    once: bool = typer.Option(False, help="Process at most one queued job and exit."),
) -> None:
    """Run the durable sequential PRIMERA/Qwen/Qwen-LoRA generation worker."""
    from related_work_assistant.generation.worker import run_worker

    settings = Settings()
    engine = make_engine(settings)
    try:
        run_worker(engine, settings, once=once)
    finally:
        engine.dispose()


@app.command("regenerate-qwen")
def regenerate_qwen(
    all_papers: bool = typer.Option(
        False, "--all", help="Required safety flag: enqueue every paper in PostgreSQL."
    ),
) -> None:
    """Force-enqueue Qwen regeneration for every paper without deleting history."""
    import json

    from related_work_assistant.generation.service import (
        bulk_regenerate_qwen,
        qwen_job_status,
    )

    if not all_papers:
        raise typer.BadParameter("Pass --all to confirm bulk Qwen regeneration")
    settings = Settings()
    engine = make_engine(settings)
    try:
        result = {
            "enqueue": bulk_regenerate_qwen(engine, settings),
            "status": qwen_job_status(engine),
        }
    finally:
        engine.dispose()
    typer.echo(json.dumps(result, indent=2))
    typer.echo("Queued only; run exactly one `uv run related-work generation-worker` process.")


@app.command("qwen-status")
def qwen_status() -> None:
    """Show aggregate Qwen queued/running/completed/failed job counts."""
    import json

    from related_work_assistant.generation.service import qwen_job_status

    engine = make_engine(Settings())
    try:
        typer.echo(json.dumps(qwen_job_status(engine), indent=2))
    finally:
        engine.dispose()


@app.command("regenerate-qwen-lora")
def regenerate_qwen_lora(
    all_papers: bool = typer.Option(
        False, "--all", help="Required safety flag: enqueue every paper in PostgreSQL."
    ),
) -> None:
    """Force-enqueue trained Qwen LoRA generation for every paper."""
    import json

    from related_work_assistant.generation.service import (
        bulk_regenerate_qwen_lora,
        qwen_lora_job_status,
    )

    if not all_papers:
        raise typer.BadParameter("Pass --all to confirm bulk Qwen LoRA generation")
    settings = Settings()
    engine = make_engine(settings)
    try:
        result = {
            "enqueue": bulk_regenerate_qwen_lora(engine, settings),
            "status": qwen_lora_job_status(engine),
        }
    finally:
        engine.dispose()
    typer.echo(json.dumps(result, indent=2))
    typer.echo("Queued only; run exactly one `uv run related-work generation-worker` process.")


@app.command("qwen-lora-generation-status")
def qwen_lora_generation_status() -> None:
    """Show aggregate Qwen LoRA generation job and result counts."""
    import json

    from related_work_assistant.generation.service import qwen_lora_job_status

    engine = make_engine(Settings())
    try:
        typer.echo(json.dumps(qwen_lora_job_status(engine), indent=2))
    finally:
        engine.dispose()


@app.command("prepare-multixscience")
def prepare_multixscience(
    limit: int | None = typer.Option(
        None, min=1, help="Process only the first N examples of each split."
    ),
    max_sequence_tokens: int = typer.Option(32768, min=1),
    tokenizer_revision: str | None = typer.Option(None, "--revision"),
) -> None:
    """Download/cache and citation-aware preprocess the official Multi-XScience splits."""
    import json

    from related_work_assistant.datasets.multixscience import (
        load_qwen_tokenizer,
        prepare_dataset,
    )

    tokenizer = load_qwen_tokenizer(revision=tokenizer_revision)
    result = prepare_dataset(
        tokenizer=tokenizer,
        tokenizer_revision=tokenizer_revision,
        limit=limit,
        max_sequence_tokens=max_sequence_tokens,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("inspect-multixscience")
def inspect_multixscience(
    split: str = typer.Option("train", help="train, validation, or test"),
    limit: int = typer.Option(1, min=1, max=100),
) -> None:
    """Print prepared examples without loading an entire JSONL split."""
    import json

    from related_work_assistant.datasets.multixscience import inspect_examples

    try:
        rows = inspect_examples(split=split, limit=limit)
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))


@app.command("analyze-multixscience-invalid")
def analyze_multixscience_invalid() -> None:
    """Explain invalid preprocessing results and audit exact-mid recovery potential."""
    import json

    from related_work_assistant.datasets.multixscience import analyze_invalid_dataset

    try:
        statistics = analyze_invalid_dataset()
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(statistics, ensure_ascii=False, indent=2))


@app.command("train-qwen-lora-smoke")
def train_qwen_lora_smoke(
    train_limit: int = typer.Option(8, min=1),
    validation_limit: int = typer.Option(4, min=1),
    max_steps: int = typer.Option(2, min=1),
    device: str = typer.Option("auto", help="auto, cpu, cuda, or mps"),
    mixed_precision: str = typer.Option("auto", help="auto, none, fp16, or bf16"),
) -> None:
    """Run a tiny isolated LoRA training that cannot activate production inference."""
    import json

    from related_work_assistant.training.qwen_lora import (
        QwenLoraTrainingConfig,
        run_qwen_lora_training,
    )

    if device not in {"auto", "cpu", "cuda", "mps"}:
        raise typer.BadParameter("--device must be auto, cpu, cuda, or mps")
    if mixed_precision not in {"auto", "none", "fp16", "bf16"}:
        raise typer.BadParameter("--mixed-precision must be auto, none, fp16, or bf16")
    config = QwenLoraTrainingConfig.smoke(
        train_limit=train_limit,
        validation_limit=validation_limit,
        max_steps=max_steps,
        device=device,
        mixed_precision=mixed_precision,
    )
    typer.echo(json.dumps(run_qwen_lora_training(config), indent=2, default=str))


@app.command("train-qwen-lora")
def train_qwen_lora(
    epochs: float = typer.Option(3.0, min=0.1),
    batch_size: int = typer.Option(1, min=1),
    gradient_accumulation: int = typer.Option(16, min=1),
    learning_rate: float = typer.Option(2e-4, min=1e-8),
    device: str = typer.Option("auto", help="auto, cpu, cuda, or mps"),
    mixed_precision: str = typer.Option("auto", help="auto, none, fp16, or bf16"),
    resume: str | None = typer.Option(
        None, help="Trainer checkpoint path, or 'latest' for the newest checkpoint."
    ),
) -> None:
    """Fine-tune the production Qwen LoRA adapter on train with validation loss."""
    import json

    from related_work_assistant.training.lora_artifacts import QWEN_LORA_ROOT
    from related_work_assistant.training.qwen_lora import (
        QwenLoraTrainingConfig,
        run_qwen_lora_training,
    )

    if device not in {"auto", "cpu", "cuda", "mps"}:
        raise typer.BadParameter("--device must be auto, cpu, cuda, or mps")
    if mixed_precision not in {"auto", "none", "fp16", "bf16"}:
        raise typer.BadParameter("--mixed-precision must be auto, none, fp16, or bf16")
    resume_path = resume
    if resume == "latest":
        candidates = sorted(
            (QWEN_LORA_ROOT / "checkpoints").glob("checkpoint-*"),
            key=lambda path: int(path.name.rsplit("-", 1)[-1]),
        )
        if not candidates:
            raise typer.BadParameter("No Qwen LoRA checkpoints are available to resume")
        resume_path = str(candidates[-1])
    config = QwenLoraTrainingConfig(
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        device=device,
        mixed_precision=mixed_precision,
        resume_from_checkpoint=resume_path,
    )
    typer.echo(json.dumps(run_qwen_lora_training(config), indent=2, default=str))


@app.command("qwen-lora-training-status")
def qwen_lora_training_status(
    smoke: bool = typer.Option(False, help="Inspect isolated smoke-test artifacts."),
) -> None:
    """Show adapter availability, checkpoints, configuration and saved metrics."""
    import json

    from related_work_assistant.training.lora_artifacts import (
        QWEN_LORA_ROOT,
        QWEN_LORA_SMOKE_ROOT,
        training_artifact_status,
    )

    root = QWEN_LORA_SMOKE_ROOT if smoke else QWEN_LORA_ROOT
    typer.echo(
        json.dumps(training_artifact_status(root), ensure_ascii=False, indent=2, default=str)
    )


@app.command("evaluate-generation-a")
def evaluate_generation_a(
    limit: int = typer.Option(10, min=1, help="Reproducible sample size."),
    seed: int = typer.Option(42),
    full: bool = typer.Option(False, "--full", help="Evaluate all 3,209 prepared test examples."),
    device: str = typer.Option("auto", help="auto, cpu, cuda, or mps"),
) -> None:
    """Run the abstract-only Multi-XScience ROUGE benchmark explicitly."""
    import json

    from related_work_assistant.evaluation.generation.benchmark import run_evaluation_a

    if device not in {"auto", "cpu", "cuda", "mps"}:
        raise typer.BadParameter("--device must be auto, cpu, cuda, or mps")
    result = run_evaluation_a(limit=None if full else limit, seed=seed, device=device)
    summary = {key: value for key, value in result.items() if key != "examples"}
    summary["example_count"] = len(result["examples"])
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


@app.command("evaluate-generation-b")
def evaluate_generation_b(
    paper_id: str = typer.Argument(..., help="Target paper UUID."),
    device: str = typer.Option("auto", help="NLI device: auto, cpu, cuda, or mps"),
) -> None:
    """Evaluate persisted generations and source-grounded evidence for one paper."""
    import json
    from uuid import UUID

    from related_work_assistant.evaluation.generation.evidence import run_evaluation_b

    if device not in {"auto", "cpu", "cuda", "mps"}:
        raise typer.BadParameter("--device must be auto, cpu, cuda, or mps")
    try:
        parsed_id = UUID(paper_id)
    except ValueError as exc:
        raise typer.BadParameter("paper_id must be a UUID") from exc
    settings = Settings()
    engine = make_engine(settings)
    try:
        result = run_evaluation_b(engine, settings, [parsed_id], device=device)
    finally:
        engine.dispose()
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=str))


@app.command("evaluate-generation-b-fixed")
def evaluate_generation_b_fixed(
    limit: int = typer.Option(10, min=1, max=50),
    seed: int = typer.Option(42),
    reset_fixed_set: bool = typer.Option(False, help="Select and persist a new fixed paper set."),
    device: str = typer.Option("auto", help="NLI device: auto, cpu, cuda, or mps"),
) -> None:
    """Evaluate persisted outputs on a reproducible fixed target-paper set."""
    import json

    from related_work_assistant.evaluation.generation.evidence import (
        fixed_paper_ids,
        run_evaluation_b,
    )

    if device not in {"auto", "cpu", "cuda", "mps"}:
        raise typer.BadParameter("--device must be auto, cpu, cuda, or mps")
    settings = Settings()
    engine = make_engine(settings)
    try:
        paper_ids = fixed_paper_ids(
            engine, limit=limit, seed=seed, reset=reset_fixed_set
        )
        result = run_evaluation_b(
            engine, settings, paper_ids, device=device, fixed_set=True
        )
    finally:
        engine.dispose()
    summary = {key: value for key, value in result.items() if key != "papers"}
    summary["paper_count"] = len(result["papers"])
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


@app.command("generation-evaluation-results")
def generation_evaluation_results(
    kind: str = typer.Option("a", help="a or b"),
    limit: int = typer.Option(5, min=1, max=50),
) -> None:
    """List saved versioned Evaluation A or B run summaries."""
    import json

    from related_work_assistant.evaluation.generation.storage import list_runs

    if kind not in {"a", "b"}:
        raise typer.BadParameter("--kind must be a or b")
    runs = list_runs(f"evaluation_{kind}", limit=limit)
    summaries = []
    for run in runs:
        summary = {key: value for key, value in run.items() if key not in {"examples", "papers"}}
        summary["item_count"] = len(run.get("examples", run.get("papers", [])))
        summaries.append(summary)
    typer.echo(json.dumps(summaries, ensure_ascii=False, indent=2, default=str))


@app.command("chunk")
def chunk_papers(
    path: str = typer.Argument(
        "data/raw_data/papers", help="One ingested PDF or a directory below RAW_PDFS_DIR."
    ),
    limit: int | None = typer.Option(None, min=1, help="Maximum unique PDFs to process."),
    min_tokens: int | None = typer.Option(None, min=1),
    target_tokens: int | None = typer.Option(None, min=1),
    max_tokens: int | None = typer.Option(None, min=1),
    force: bool = typer.Option(
        False, help="Re-run GROBID full-text and transactionally replace sections/chunks."
    ),
) -> None:
    """Create section-aware chunks and map them to PyMuPDF word coordinates."""
    from pathlib import Path

    from pydantic import ValidationError

    from related_work_assistant.chunking.section_aware import ChunkingConfig
    from related_work_assistant.extraction.grobid import GrobidClient
    from related_work_assistant.pipelines.chunking import (
        process_paper_file,
        select_paper_files,
    )
    from related_work_assistant.pipelines.ingestion import write_json

    settings = Settings()
    try:
        config = ChunkingConfig(
            min_tokens=min_tokens or settings.chunk_min_tokens,
            target_tokens=target_tokens or settings.chunk_target_tokens,
            max_tokens=max_tokens or settings.chunk_max_tokens,
        )
    except ValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    target = (settings.storage_root / Path(path)).resolve()
    if not target.exists():
        raise typer.BadParameter("Path does not exist")
    engine = make_engine(settings)
    results = []
    try:
        try:
            file_ids = select_paper_files(engine, settings, target)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        if limit:
            file_ids = file_ids[:limit]
        if not file_ids:
            raise typer.BadParameter("No ingested PDFs match this path")
        with GrobidClient(settings.grobid_url, settings.grobid_timeout) as client:
            for file_id in file_ids:
                try:
                    result = process_paper_file(
                        file_id, settings, engine, client, config, force=force
                    )
                except Exception as exc:
                    result = {
                        "status": "failed",
                        "paper_file_id": str(file_id),
                        "error": str(exc)
                        if isinstance(exc, (ValueError, RuntimeError))
                        else type(exc).__name__,
                    }
                results.append(result)
                typer.echo(
                    f"{result['status']}: {result.get('path', file_id)} "
                    f"({result.get('section_count', 0)} sections, "
                    f"{result.get('chunk_count', 0)} chunks)"
                )
                write_json(settings.processed_dir / "chunking_manifest.json", results)
    finally:
        engine.dispose()
    output = settings.processed_dir / "chunking_manifest.json"
    typer.echo(f"Manifest: {output}")
    if any(result["status"] == "failed" for result in results):
        raise typer.Exit(1)


@app.command("embed-papers")
def embed_paper_vectors(
    paper_id: list[str] | None = typer.Option(
        None, "--paper-id", help="Paper UUID; repeat to process several. Omit for all papers."
    ),
    batch_size: int = typer.Option(8, min=1),
    device: str = typer.Option("auto", help="auto, mps, cuda, or cpu"),
    revision: str | None = typer.Option(None, help="SPECTER2 base revision/tag/SHA."),
    adapter_revision: str | None = typer.Option(None, help="SPECTER2 adapter revision/tag/SHA."),
    limit: int | None = typer.Option(None, min=1, hidden=True),
) -> None:
    """Embed GROBID title + abstract with SPECTER2 retrieval adapter."""
    from related_work_assistant.embeddings.encoders import Specter2PaperEncoder
    from related_work_assistant.pipelines.embeddings import embed_papers
    from related_work_assistant.pipelines.ingestion import write_json

    settings = Settings()
    engine = make_engine(settings)
    try:
        encoder = Specter2PaperEncoder(
            settings.model_cache_dir,
            device=device,
            revision=revision,
            adapter_revision=adapter_revision,
        )
        result = embed_papers(
            engine,
            encoder,
            encoder.separator,
            paper_ids=_paper_ids(paper_id),
            batch_size=batch_size,
            limit=limit,
        )
        output = settings.processed_dir / "paper_embeddings_manifest.json"
        write_json(output, result.as_dict())
    finally:
        engine.dispose()
    typer.echo(
        f"Paper embeddings: {result.embedded} written, {result.updated} updated, "
        f"{result.skipped} unchanged, {result.ineligible} missing title/abstract."
    )
    typer.echo(f"Model: {result.model_key}")
    typer.echo(f"Manifest: {output}")


@app.command("embed-chunks")
def embed_chunk_vectors(
    paper_id: list[str] | None = typer.Option(
        None, "--paper-id", help="Paper UUID; repeat to process several. Omit for all chunks."
    ),
    batch_size: int = typer.Option(32, min=1),
    device: str = typer.Option("auto", help="auto, mps, cuda, or cpu"),
    revision: str | None = typer.Option(None, help="BGE checkpoint revision/tag/SHA."),
    limit: int | None = typer.Option(None, min=1, hidden=True),
) -> None:
    """Embed existing chunks with BGE-small-en-v1.5."""
    from related_work_assistant.embeddings.encoders import BgeChunkEncoder
    from related_work_assistant.pipelines.embeddings import embed_chunks
    from related_work_assistant.pipelines.ingestion import write_json

    settings = Settings()
    engine = make_engine(settings)
    try:
        encoder = BgeChunkEncoder(
            settings.model_cache_dir,
            device=device,
            revision=revision,
        )
        result = embed_chunks(
            engine,
            encoder,
            paper_ids=_paper_ids(paper_id),
            batch_size=batch_size,
            limit=limit,
        )
        output = settings.processed_dir / "chunk_embeddings_manifest.json"
        write_json(output, result.as_dict())
    finally:
        engine.dispose()
    typer.echo(
        f"Chunk embeddings: {result.embedded} written, {result.updated} updated, "
        f"{result.skipped} unchanged."
    )
    typer.echo(f"Model: {result.model_key}")
    typer.echo(f"Manifest: {output}")


@app.command("embeddings-status")
def show_embeddings_status() -> None:
    """Show registered embedding revisions, dimensions, and persisted row counts."""
    import json

    from related_work_assistant.pipelines.embeddings import embedding_status

    engine = make_engine(Settings())
    try:
        typer.echo(json.dumps(embedding_status(engine), indent=2))
    finally:
        engine.dispose()
