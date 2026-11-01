"""Extract source text and geometry without OCR or rewriting the text."""

from pathlib import Path

import pymupdf

from related_work_assistant.extraction.schemas import PageText, PDFText, TextBlock, Word


class PDFExtractionError(ValueError):
    pass


def extract_pdf(path: Path) -> PDFText:
    try:
        document = pymupdf.open(path)
    except Exception as exc:
        raise PDFExtractionError("Cannot open PDF") from exc
    with document:
        if not document.is_pdf or document.needs_pass:
            raise PDFExtractionError("Input must be an unencrypted PDF")
        if not document.page_count:
            raise PDFExtractionError("PDF has no pages")
        pages = []
        warnings = []
        for number, page in enumerate(document, start=1):
            blocks = []
            offset = 0
            for x0, y0, x1, y1, content, block_index, block_type in page.get_text(
                "blocks", sort=False
            ):
                if block_type != 0:
                    continue
                blocks.append(
                    TextBlock(
                        block_index=block_index,
                        bbox=(x0, y0, x1, y1),
                        text=content,
                        char_start=offset,
                        char_end=offset + len(content),
                    )
                )
                offset += len(content)
            text = "".join(block.text for block in blocks)
            words = [
                Word(
                    bbox=tuple(w[:4]), text=w[4], block_index=w[5], line_index=w[6], word_index=w[7]
                )
                for w in page.get_text("words", sort=False)
            ]
            if not text.strip():
                warnings.append(f"Page {number}: no selectable text; OCR not performed")
            pages.append(
                PageText(
                    page_number=number,
                    width=page.cropbox.width,
                    height=page.cropbox.height,
                    rotation=page.rotation,
                    text=text,
                    blocks=blocks,
                    words=words,
                )
            )
        if not any(p.text.strip() for p in pages):
            raise PDFExtractionError("PDF has no selectable text; OCR is required")
        return PDFText(parser_version=pymupdf.VersionBind, pages=pages, warnings=warnings)
