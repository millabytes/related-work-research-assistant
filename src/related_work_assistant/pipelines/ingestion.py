"""Idempotent local PDF ingestion, with durable audit records and raw artifacts."""

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from uuid import UUID

from sqlalchemy import Engine, select, text

from related_work_assistant.core.config import Settings
from related_work_assistant.database.models import IngestionRun, Paper, PaperAuthor, PaperFile
from related_work_assistant.database.session import session_scope
from related_work_assistant.extraction.grobid import GrobidClient
from related_work_assistant.extraction.pdf import extract_pdf
from related_work_assistant.extraction.schemas import PaperMetadata


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def metadata_from_paper(paper: Paper, authors: list[PaperAuthor]) -> PaperMetadata:
    return PaperMetadata(
        title=paper.title,
        authors=[{"first_name": a.first_name, "last_name": a.last_name} for a in authors],
        year=paper.publication_year,
        venue=paper.venue_name,
        doi=paper.doi,
        abstract=paper.abstract,
        keywords=paper.keywords,
    )


def ingest_pdf(
    path: Path, settings: Settings, engine: Engine, grobid: GrobidClient, force: bool = False
) -> dict:
    path = path.resolve()
    try:
        relative = path.relative_to(settings.storage_root).as_posix()
    except ValueError as exc:
        raise ValueError("PDF must be inside STORAGE_ROOT") from exc
    if path.suffix.lower() != ".pdf" or not path.is_file():
        raise ValueError("Input must be an existing PDF file")
    digest = sha256(path.read_bytes()).hexdigest()
    with session_scope(engine) as session:
        file = session.scalar(select(PaperFile).where(PaperFile.sha256 == digest))
        if file is not None and not force:
            run = session.scalar(
                select(IngestionRun)
                .where(IngestionRun.paper_file_id == file.id, IngestionRun.status == "succeeded")
                .order_by(IngestionRun.finished_at.desc())
                .limit(1)
            )
            if (
                run
                and (settings.storage_root / run.config["artifact_dir"] / "metadata.json").exists()
            ):
                return {
                    "path": relative,
                    "sha256": digest,
                    "status": "skipped",
                    "paper_id": str(file.paper_id),
                    "run_id": str(run.id),
                    "artifact_dir": run.config["artifact_dir"],
                }
        run = IngestionRun(
            status="running",
            stage="text",
            parser_name="pymupdf+grobid",
            paper_file_id=file.id if file else None,
            config={"relative_path": relative, "sha256": digest, "consolidateHeader": 0},
        )
        session.add(run)
        session.flush()
        run_id = run.id
    artifact_dir = settings.processed_dir / digest / str(run_id)
    stage = "text"
    try:
        document = extract_pdf(path)
        write_json(artifact_dir / "text.json", document.model_dump(mode="json"))
        (artifact_dir / "text.txt").write_text(
            "\n\n".join(f"--- Page {p.page_number} ---\n{p.text}" for p in document.pages)
        )
        stage = "metadata"
        with session_scope(engine) as session:
            session.get(IngestionRun, run_id).stage = stage
        version = grobid.version()
        metadata, tei = grobid.extract(path)
        (artifact_dir / "header.tei.xml").write_text(tei)
        write_json(artifact_dir / "metadata.json", metadata.model_dump(mode="json"))
        warnings = list(document.warnings)
        if not metadata.title:
            warnings.append("GROBID did not identify a title; stored as null")
        stage = "persist"
        with session_scope(engine) as session:
            # Serialize writes for an exact file and shared DOI; extraction stays outside transactions.
            session.execute(
                text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"), {"key": digest}
            )
            if metadata.doi:
                session.execute(
                    text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
                    {"key": metadata.doi},
                )
            file = session.scalar(select(PaperFile).where(PaperFile.sha256 == digest))
            paper = session.get(Paper, file.paper_id) if file else None
            if paper is None and metadata.doi:
                paper = session.scalar(select(Paper).where(Paper.doi == metadata.doi))
            artifact_relative = artifact_dir.relative_to(settings.storage_root).as_posix()
            if paper is None:
                paper = Paper(
                    title=metadata.title,
                    abstract=metadata.abstract,
                    publication_year=metadata.year,
                    venue_name=metadata.venue,
                    doi=metadata.doi,
                    keywords=metadata.keywords,
                    metadata_source="grobid",
                    metadata_extra={"grobid_version": version, "artifact_dir": artifact_relative},
                )
                session.add(paper)
                session.flush()
                for position, author in enumerate(metadata.authors, start=1):
                    if author.full_name:
                        session.add(
                            PaperAuthor(
                                paper_id=paper.id,
                                author_position=position,
                                full_name=author.full_name,
                                first_name=author.first_name,
                                last_name=author.last_name,
                            )
                        )
            else:
                warnings.append(
                    "Existing paper retained; new extraction preserved separately for review"
                )
            if file is None:
                file = PaperFile(
                    paper_id=paper.id,
                    relative_path=relative,
                    sha256=digest,
                    byte_size=path.stat().st_size,
                    page_count=len(document.pages),
                )
                session.add(file)
                session.flush()
            run = session.get(IngestionRun, run_id)
            run.paper_file_id = file.id
            run.status = "succeeded"
            run.stage = "complete"
            run.parser_version = f"pymupdf:{document.parser_version};grobid:{version}"
            run.warnings = warnings
            run.finished_at = datetime.now(UTC)
            run.config = {
                **run.config,
                "artifact_dir": artifact_relative,
                "grobid_version": version,
                "metadata": metadata.model_dump(mode="json"),
            }
            result = {
                "path": relative,
                "sha256": digest,
                "status": "succeeded",
                "paper_id": str(paper.id),
                "run_id": str(run_id),
                "artifact_dir": artifact_relative,
            }
        return result
    except Exception as exc:
        with session_scope(engine) as session:
            run = session.get(IngestionRun, run_id)
            run.status = "failed"
            run.stage = stage
            # Do not persist driver exception strings containing SQL parameters or secrets.
            run.error_message = f"{type(exc).__name__}: failure during {stage}"
            run.finished_at = datetime.now(UTC)
        raise RuntimeError(f"Ingestion failed during {stage}; audit run {run_id}") from exc


def load_metadata(engine: Engine, paper_id: UUID) -> PaperMetadata:
    with session_scope(engine) as session:
        paper = session.get(Paper, paper_id)
        if paper is None:
            raise ValueError("Unknown paper ID")
        authors = list(
            session.scalars(
                select(PaperAuthor)
                .where(PaperAuthor.paper_id == paper_id)
                .order_by(PaperAuthor.author_position)
            )
        )
        return metadata_from_paper(paper, authors)
