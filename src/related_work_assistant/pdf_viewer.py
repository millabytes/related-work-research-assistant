"""Read-only PDF lookup and in-memory highlighted page rendering."""

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import pymupdf
from sqlalchemy import Engine, select

from related_work_assistant.core.config import Settings
from related_work_assistant.database.models import Paper, PaperFile
from related_work_assistant.database.session import session_scope


class PDFNotFoundError(LookupError):
    pass


@dataclass(frozen=True)
class PDFInfo:
    path: Path
    filename: str
    page_count: int


def paper_pdf_info(engine: Engine, settings: Settings, paper_id: UUID) -> PDFInfo:
    with session_scope(engine) as session:
        if session.get(Paper, paper_id) is None:
            raise PDFNotFoundError(f"Paper {paper_id} was not found")
        file = session.scalar(
            select(PaperFile)
            .where(PaperFile.paper_id == paper_id)
            .order_by(PaperFile.created_at.desc(), PaperFile.id.desc())
            .limit(1)
        )
    if file is None:
        raise PDFNotFoundError(f"Paper {paper_id} has no PDF file")
    path = (settings.storage_root / file.relative_path).resolve()
    if not path.is_relative_to(settings.raw_pdfs_dir) or not path.is_file():
        raise PDFNotFoundError(f"PDF for paper {paper_id} is missing")
    return PDFInfo(path=path, filename=path.name, page_count=file.page_count or 0)


def _rgb(value: str) -> tuple[float, float, float]:
    normalized = value.removeprefix("#")
    if len(normalized) != 6:
        raise ValueError("Highlight color must be a six-digit hex value")
    try:
        channels = tuple(int(normalized[index : index + 2], 16) / 255 for index in (0, 2, 4))
    except ValueError as exc:
        raise ValueError("Highlight color must be a six-digit hex value") from exc
    return channels


def render_pdf_page(
    info: PDFInfo,
    page_number: int,
    highlights: list[dict],
    *,
    zoom: float = 1.6,
) -> bytes:
    with pymupdf.open(info.path) as document:
        if page_number < 1 or page_number > document.page_count:
            raise ValueError(f"Page must be between 1 and {document.page_count}")
        page = document.load_page(page_number - 1)
        for highlight in highlights:
            bbox = highlight.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ValueError("Each highlight requires a four-number bbox")
            rect = pymupdf.Rect(*(float(value) for value in bbox))
            if rect.is_empty or rect.is_infinite or not page.rect.contains(rect):
                raise ValueError("Highlight bbox must be inside the selected PDF page")
            annotation = page.add_highlight_annot(rect)
            annotation.set_colors(stroke=_rgb(str(highlight.get("color", "#FFD54F"))))
            annotation.set_opacity(0.38)
            annotation.update()
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False, annots=True)
        return pixmap.tobytes("png")
