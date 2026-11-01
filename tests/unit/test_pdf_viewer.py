from hashlib import sha256

import pymupdf
import pytest

from related_work_assistant.pdf_viewer import PDFInfo, render_pdf_page


def test_pdf_page_render_is_in_memory_and_preserves_original(tmp_path):
    path = tmp_path / "paper.pdf"
    with pymupdf.open() as document:
        page = document.new_page(width=300, height=400)
        page.insert_text((30, 80), "Mapped scientific sentence.")
        document.save(path)
    before = sha256(path.read_bytes()).hexdigest()
    image = render_pdf_page(
        PDFInfo(path=path, filename=path.name, page_count=1),
        1,
        [{"bbox": [25, 65, 220, 85], "color": "#FFD54F"}],
    )
    assert image.startswith(b"\x89PNG")
    assert sha256(path.read_bytes()).hexdigest() == before


def test_pdf_render_rejects_fabricated_out_of_page_coordinates(tmp_path):
    path = tmp_path / "paper.pdf"
    with pymupdf.open() as document:
        document.new_page(width=300, height=400)
        document.save(path)
    info = PDFInfo(path=path, filename=path.name, page_count=1)
    with pytest.raises(ValueError, match="inside"):
        render_pdf_page(info, 1, [{"bbox": [-5, 10, 100, 20], "color": "#64B5F6"}])
