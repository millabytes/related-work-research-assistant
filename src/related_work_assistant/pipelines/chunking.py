"""Cached GROBID full-text extraction and transactional section/chunk persistence."""

import json
from hashlib import sha256
from pathlib import Path
from uuid import UUID

from sqlalchemy import Engine, delete, func, select, text

from related_work_assistant.chunking.section_aware import (
    ChunkingConfig,
    ChunkingResult,
    build_chunks,
)
from related_work_assistant.core.config import Settings
from related_work_assistant.database.models import (
    Chunk,
    IngestionRun,
    PaperFile,
    PaperSection,
)
from related_work_assistant.database.session import session_scope
from related_work_assistant.extraction.fulltext import StructuredDocument, parse_fulltext
from related_work_assistant.extraction.grobid import GrobidClient
from related_work_assistant.extraction.schemas import PDFText
from related_work_assistant.pipelines.ingestion import write_json

FULLTEXT_REQUEST = {
    "endpoint": "processFulltextDocument",
    "consolidateHeader": 0,
    "consolidateCitations": 0,
    "segmentSentences": 1,
    "generateIDs": 1,
    "teiCoordinates": ["s", "p", "head"],
}


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def _artifact_dir(engine: Engine, file_id: UUID, settings: Settings) -> Path:
    with session_scope(engine) as session:
        run = session.scalar(
            select(IngestionRun)
            .where(IngestionRun.paper_file_id == file_id, IngestionRun.status == "succeeded")
            .order_by(IngestionRun.finished_at.desc())
            .limit(1)
        )
        if run is None or not run.config.get("artifact_dir"):
            raise ValueError("PDF has no successful ingestion artifact")
        artifact_dir = (settings.storage_root / run.config["artifact_dir"]).resolve()
    if not artifact_dir.is_relative_to(settings.processed_dir):
        raise ValueError("Ingestion artifact is outside PROCESSED_DIR")
    return artifact_dir


def _load_or_extract_fulltext(
    pdf: Path,
    artifact_dir: Path,
    pdf_sha256: str,
    grobid: GrobidClient,
    force: bool,
) -> tuple[str, str, bool]:
    tei_path = artifact_dir / "fulltext.tei.xml"
    manifest_path = artifact_dir / "fulltext.manifest.json"
    version = grobid.version()
    expected = {
        "pdf_sha256": pdf_sha256,
        "grobid_version": version,
        "request": FULLTEXT_REQUEST,
    }
    if not force and tei_path.exists() and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            manifest = None
        if manifest == expected:
            return tei_path.read_text(), version, True
    xml = grobid.extract_fulltext(pdf)
    # Parse before replacing a valid cache with a malformed response.
    parse_fulltext(xml)
    _write_text(tei_path, xml)
    write_json(manifest_path, expected)
    return xml, version, False


def _page_range(result: ChunkingResult, section_order: int) -> tuple[int | None, int | None]:
    pages = [
        page
        for chunk in result.chunks
        if chunk.section_order == section_order
        for page in (chunk.page_start, chunk.page_end)
        if page is not None
    ]
    return (min(pages), max(pages)) if pages else (None, None)


def _persist(
    engine: Engine,
    file_id: UUID,
    paper_id: UUID,
    source_sha256: str,
    parser_version: str,
    document: StructuredDocument,
    result: ChunkingResult,
    force: bool,
) -> str:
    with session_scope(engine) as session:
        session.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
            {"key": f"chunk:{paper_id}"},
        )
        existing_sections = list(
            session.scalars(select(PaperSection).where(PaperSection.paper_id == paper_id))
        )
        versions = set(
            session.scalars(select(Chunk.chunker_version).where(Chunk.paper_id == paper_id))
        )
        existing_chunk_count = session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.paper_id == paper_id)
        )
        unchanged = (
            bool(existing_sections)
            and {section.source_sha256 for section in existing_sections} == {source_sha256}
            and versions == {result.version}
            and existing_chunk_count == len(result.chunks)
        )
        if unchanged and not force:
            return "skipped"

        session.execute(delete(PaperSection).where(PaperSection.paper_id == paper_id))
        session.flush()
        section_rows = {}
        for order, section in enumerate(document.sections):
            page_start, page_end = _page_range(result, order)
            row = PaperSection(
                paper_id=paper_id,
                source_file_id=file_id,
                source_sha256=source_sha256,
                parser_version=parser_version,
                section_order=order,
                heading=section.heading,
                section_type=section.section_type,
                page_start=page_start,
                page_end=page_end,
            )
            session.add(row)
            session.flush()
            section_rows[order] = row

        for index, chunk in enumerate(result.chunks):
            session.add(
                Chunk(
                    paper_id=paper_id,
                    section_id=section_rows[chunk.section_order].id,
                    chunk_index=index,
                    chunk_type="abstract" if chunk.section_type == "abstract" else "body",
                    content=chunk.text,
                    content_sha256=chunk.content_hash,
                    token_count=chunk.token_count,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    source_spans=chunk.source_mapping,
                    chunker_version=result.version,
                )
            )
    return "succeeded"


def process_paper_file(
    file_id: UUID,
    settings: Settings,
    engine: Engine,
    grobid: GrobidClient,
    config: ChunkingConfig,
    force: bool = False,
) -> dict:
    with session_scope(engine) as session:
        file = session.get(PaperFile, file_id)
        if file is None:
            raise ValueError("Unknown paper file ID")
        paper_id = file.paper_id
        relative_path = file.relative_path
        pdf_sha256 = file.sha256
    pdf = (settings.storage_root / relative_path).resolve()
    if not pdf.is_relative_to(settings.raw_pdfs_dir) or not pdf.is_file():
        raise ValueError("Stored PDF path is missing or outside RAW_PDFS_DIR")
    artifact_dir = _artifact_dir(engine, file_id, settings)
    source_path = artifact_dir / "text.json"
    if not source_path.is_file():
        raise ValueError("PyMuPDF text.json artifact is missing")
    source = PDFText.model_validate_json(source_path.read_text())
    xml, grobid_version, cache_hit = _load_or_extract_fulltext(
        pdf, artifact_dir, pdf_sha256, grobid, force
    )
    document = parse_fulltext(xml)
    result = build_chunks(document, source, config)
    source_sha256 = sha256((xml + "\0" + source_path.read_text()).encode()).hexdigest()
    parser_version = f"grobid:{grobid_version};pymupdf:{source.parser_version}"
    status = _persist(
        engine,
        file_id,
        paper_id,
        source_sha256,
        parser_version,
        document,
        result,
        force,
    )
    write_json(
        artifact_dir / "chunking.json",
        {
            "source_sha256": source_sha256,
            "chunking_version": result.version,
            "config": config.model_dump(),
            "sections": [section.model_dump(mode="json") for section in document.sections],
            "chunks": [chunk.model_dump(mode="json") for chunk in result.chunks],
            "warnings": result.warnings,
        },
    )
    return {
        "status": status,
        "paper_id": str(paper_id),
        "paper_file_id": str(file_id),
        "path": relative_path,
        "artifact_dir": artifact_dir.relative_to(settings.storage_root).as_posix(),
        "fulltext_cache_hit": cache_hit,
        "section_count": len(document.sections),
        "chunk_count": len(result.chunks),
        "chunking_version": result.version,
        "token_counts": [chunk.token_count for chunk in result.chunks],
        "mapping": {
            status: sum(
                mapping["mapping_status"] == status
                for chunk in result.chunks
                for mapping in chunk.source_mapping
            )
            for status in ("mapped", "weak", "unmapped")
        },
        "examples": [
            {
                "section": chunk.heading,
                "section_type": chunk.section_type,
                "tokens": chunk.token_count,
                "pages": [chunk.page_start, chunk.page_end],
                "text": chunk.text[:500],
            }
            for chunk in result.chunks[:2]
        ],
    }


def select_paper_files(engine: Engine, settings: Settings, target: Path) -> list[UUID]:
    target = target.resolve()
    if not target.is_relative_to(settings.raw_pdfs_dir):
        raise ValueError("Target must be inside RAW_PDFS_DIR")
    with session_scope(engine) as session:
        rows = list(session.scalars(select(PaperFile).order_by(PaperFile.relative_path)))
        selected = []
        for row in rows:
            candidate = (settings.storage_root / row.relative_path).resolve()
            if candidate == target or (target.is_dir() and candidate.is_relative_to(target)):
                selected.append(row.id)
        return selected
