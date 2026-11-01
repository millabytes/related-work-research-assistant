"""Deterministic section-aware and paragraph-aware chunk construction."""

import json
import re
from hashlib import sha256

from pydantic import BaseModel, Field, model_validator

from related_work_assistant.extraction.fulltext import (
    StructuredDocument,
    StructuredParagraph,
    StructuredSentence,
    map_sentence,
)
from related_work_assistant.extraction.schemas import PDFText

ALGORITHM_VERSION = "section-paragraph-v1.1"
TOKENIZER_VERSION = "unicode-regex-v1"
TOKEN_PATTERN = re.compile(r"\w+(?:[’'-]\w+)*|[^\w\s]", re.UNICODE)


class ChunkingConfig(BaseModel):
    min_tokens: int = Field(default=200, ge=1)
    target_tokens: int = Field(default=300, ge=1)
    max_tokens: int = Field(default=400, ge=1)

    @model_validator(mode="after")
    def ordered(self) -> "ChunkingConfig":
        if not self.min_tokens <= self.target_tokens <= self.max_tokens:
            raise ValueError("min_tokens must be <= target_tokens <= max_tokens")
        return self

    @property
    def version(self) -> str:
        canonical = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        digest = sha256(canonical.encode()).hexdigest()[:12]
        return f"{ALGORITHM_VERSION}:{TOKENIZER_VERSION}:{digest}"


class SectionChunk(BaseModel):
    section_order: int
    section_type: str
    heading: str | None
    text: str
    token_count: int
    content_hash: str
    page_start: int | None
    page_end: int | None
    source_mapping: list[dict]
    warnings: list[str] = Field(default_factory=list)


class ChunkingResult(BaseModel):
    chunks: list[SectionChunk]
    version: str
    warnings: list[str] = Field(default_factory=list)


def count_tokens(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


def _split_paragraph(
    paragraph: StructuredParagraph, max_tokens: int
) -> list[list[StructuredSentence]]:
    groups: list[list[StructuredSentence]] = []
    current: list[StructuredSentence] = []
    current_count = 0
    for sentence in paragraph.sentences:
        size = count_tokens(sentence.text)
        if current and current_count + size > max_tokens:
            groups.append(current)
            current = []
            current_count = 0
        current.append(sentence)
        current_count += size
        if size > max_tokens:
            groups.append(current)
            current = []
            current_count = 0
    if current:
        groups.append(current)
    return groups


def _pages(mapping: list[dict]) -> list[int]:
    return sorted(
        {region["page"] for sentence in mapping for region in sentence.get("pymupdf_regions", [])}
        or {
            coordinate["page"]
            for sentence in mapping
            for coordinate in sentence.get("grobid_coordinates", [])
        }
    )


def build_chunks(
    document: StructuredDocument, source: PDFText, config: ChunkingConfig
) -> ChunkingResult:
    chunks: list[SectionChunk] = []
    warnings: list[str] = []
    for section_order, section in enumerate(document.sections):
        units = []
        for paragraph in section.paragraphs:
            for sentence_group in _split_paragraph(paragraph, config.max_tokens):
                text = " ".join(sentence.text for sentence in sentence_group)
                units.append((text, sentence_group))

        grouped: list[list[tuple[str, list[StructuredSentence]]]] = []
        current = []
        current_count = 0
        for unit in units:
            size = count_tokens(unit[0])
            if current and (
                current_count + size > config.max_tokens
                or (
                    current_count >= config.min_tokens
                    and current_count + size > config.target_tokens
                )
            ):
                grouped.append(current)
                current = []
                current_count = 0
            current.append(unit)
            current_count += size
            if current_count >= config.target_tokens:
                grouped.append(current)
                current = []
                current_count = 0
        if current:
            if grouped:
                previous_count = sum(count_tokens(unit[0]) for unit in grouped[-1])
                if previous_count + current_count <= config.max_tokens:
                    grouped[-1].extend(current)
                else:
                    grouped.append(current)
            else:
                grouped.append(current)

        for group in grouped:
            text = "\n\n".join(unit[0] for unit in group)
            token_count = count_tokens(text)
            sentence_mappings = [
                map_sentence(sentence, source, document.page_sizes)
                for _, sentences in group
                for sentence in sentences
            ]
            pages = _pages(sentence_mappings)
            chunk_warnings = []
            if token_count < config.min_tokens:
                chunk_warnings.append("short_section_or_remainder")
            if token_count > config.max_tokens:
                chunk_warnings.append("single_sentence_exceeds_max_tokens")
            if any(mapping["mapping_status"] == "unmapped" for mapping in sentence_mappings):
                chunk_warnings.append("contains_unmapped_sentence")
            if any(mapping["mapping_status"] == "weak" for mapping in sentence_mappings):
                chunk_warnings.append("contains_weak_mapping")
            chunks.append(
                SectionChunk(
                    section_order=section_order,
                    section_type=section.section_type,
                    heading=section.heading,
                    text=text,
                    token_count=token_count,
                    content_hash=sha256(text.encode()).hexdigest(),
                    page_start=pages[0] if pages else None,
                    page_end=pages[-1] if pages else None,
                    source_mapping=sentence_mappings,
                    warnings=chunk_warnings,
                )
            )
    if not chunks:
        warnings.append("no_chunks_created")
    return ChunkingResult(chunks=chunks, version=config.version, warnings=warnings)
