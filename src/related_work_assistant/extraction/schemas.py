"""Shared extraction contract; unknown bibliographic values stay null."""

import re

from pydantic import BaseModel, Field, field_validator


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value.strip(), flags=re.I)
    return value.lower().strip() or None


class Author(BaseModel):
    first_name: str | None = None
    last_name: str | None = None

    @property
    def full_name(self) -> str:
        return " ".join(part for part in (self.first_name, self.last_name) if part)


class PaperMetadata(BaseModel):
    title: str | None = None
    authors: list[Author] = Field(default_factory=list)
    year: int | None = Field(default=None, ge=1800, le=2100)
    venue: str | None = None
    doi: str | None = None
    abstract: str | None = None
    keywords: list[str] = Field(default_factory=list)

    @field_validator("doi")
    @classmethod
    def doi_normalized(cls, value: str | None) -> str | None:
        return normalize_doi(value)


class TextBlock(BaseModel):
    block_index: int
    bbox: tuple[float, float, float, float]
    text: str
    char_start: int
    char_end: int


class Word(BaseModel):
    text: str
    bbox: tuple[float, float, float, float]
    block_index: int
    line_index: int
    word_index: int


class PageText(BaseModel):
    page_number: int
    width: float
    height: float
    rotation: int
    text: str
    blocks: list[TextBlock]
    words: list[Word]


class PDFText(BaseModel):
    parser: str = "pymupdf"
    parser_version: str
    coordinate_system: str = "pdf_points_top_left_unrotated"
    reading_order: str = "pdf_native_block_order"
    pages: list[PageText]
    warnings: list[str] = Field(default_factory=list)
