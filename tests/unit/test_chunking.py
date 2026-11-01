from pathlib import Path

import pytest
from pydantic import ValidationError

from related_work_assistant.chunking.section_aware import (
    ChunkingConfig,
    build_chunks,
    count_tokens,
)
from related_work_assistant.extraction.fulltext import (
    StructuredDocument,
    StructuredParagraph,
    StructuredSection,
    StructuredSentence,
    parse_coordinates,
    parse_fulltext,
)
from related_work_assistant.extraction.schemas import PageText, PDFText, Word

FIXTURE = Path(__file__).parents[1] / "fixtures/fulltext.tei.xml"


def source_document() -> PDFText:
    lines = {
        1: [
            ("A", 40, 80),
            ("short", 55, 80),
            ("abstract", 90, 80),
            ("sentence.", 150, 80),
            ("The", 40, 170),
            ("introduction", 65, 170),
            ("starts", 140, 170),
            ("with", 180, 170),
            ("a", 210, 170),
            ("mapped", 220, 170),
            ("sentence.", 270, 170),
        ],
        2: [
            ("The", 40, 110),
            ("method", 65, 110),
            ("is", 110, 110),
            ("evaluated", 125, 110),
            ("independently", 185, 110),
            ("from", 270, 110),
            ("the", 305, 110),
            ("introduction.", 330, 110),
        ],
    }
    pages = []
    for page_number in (1, 2):
        words = [
            Word(
                text=value,
                bbox=(x, y, x + max(len(value) * 6, 8), y + 12),
                block_index=0,
                line_index=index,
                word_index=0,
            )
            for index, (value, x, y) in enumerate(lines[page_number])
        ]
        pages.append(
            PageText(
                page_number=page_number,
                width=600,
                height=800,
                rotation=0,
                text=" ".join(word.text for word in words),
                blocks=[],
                words=words,
            )
        )
    return PDFText(parser_version="test", pages=pages)


def test_parse_fulltext_sections_sentences_and_reference_exclusion():
    document = parse_fulltext(FIXTURE.read_text())
    assert [section.section_type for section in document.sections] == [
        "abstract",
        "introduction",
        "methods",
    ]
    assert [section.heading for section in document.sections] == [
        "Abstract",
        "1 Introduction",
        "Methods",
    ]
    assert document.sections[1].paragraphs[0].sentences[0].sentence_id == "s2"
    assert "Excluded reference" not in str(document.model_dump())
    assert document.page_sizes[2] == (600, 800)


def test_numeric_figure_labels_are_not_body_text():
    xml = FIXTURE.read_text().replace(
        "A separate paragraph should retain its logical boundary.",
        " ".join(["0.0 0.2 0.4 0.6 Metric = Value"] * 30),
    )
    document = parse_fulltext(xml)
    assert all(
        "Metric = Value" not in paragraph.text
        for section in document.sections
        for paragraph in section.paragraphs
    )


def test_coordinate_parser_is_defensive():
    coordinates = parse_coordinates("1,10,20,30,40;bad;2,0,0,-1,4")
    assert len(coordinates) == 1
    assert coordinates[0].bbox == (10, 20, 40, 60)


def test_section_boundaries_mapping_and_stable_version():
    document = parse_fulltext(FIXTURE.read_text())
    config = ChunkingConfig(min_tokens=5, target_tokens=12, max_tokens=20)
    result = build_chunks(document, source_document(), config)
    assert len(result.chunks) >= 3
    assert {chunk.section_type for chunk in result.chunks} == {
        "abstract",
        "introduction",
        "methods",
    }
    assert all("reference" not in chunk.text.casefold() for chunk in result.chunks)
    assert all(chunk.content_hash for chunk in result.chunks)
    assert result.version == ChunkingConfig(min_tokens=5, target_tokens=12, max_tokens=20).version
    method = next(chunk for chunk in result.chunks if chunk.section_type == "methods")
    assert method.page_start == method.page_end == 2
    assert method.source_mapping[0]["mapping_status"] == "mapped"
    assert method.source_mapping[0]["pymupdf_word_refs"]
    assert all(chunk.token_count == count_tokens(chunk.text) for chunk in result.chunks)


def test_long_paragraph_splits_at_sentence_boundaries_and_never_crosses_sections():
    sentences = [
        StructuredSentence(text=f"Sentence {i} has several useful words.") for i in range(12)
    ]
    document = StructuredDocument(
        sections=[
            StructuredSection(
                heading="Methods",
                section_type="methods",
                paragraphs=[StructuredParagraph(sentences=sentences)],
            ),
            StructuredSection(
                heading="Results",
                section_type="results",
                paragraphs=[
                    StructuredParagraph(
                        sentences=[StructuredSentence(text="Results stay separate.")]
                    )
                ],
            ),
        ]
    )
    result = build_chunks(
        document, source_document(), ChunkingConfig(min_tokens=15, target_tokens=20, max_tokens=25)
    )
    assert len([chunk for chunk in result.chunks if chunk.section_type == "methods"]) > 1
    assert all(
        not ("Sentence" in chunk.text and "Results" in chunk.text) for chunk in result.chunks
    )
    assert all(chunk.token_count <= 25 for chunk in result.chunks)


def test_invalid_config():
    with pytest.raises(ValidationError):
        ChunkingConfig(min_tokens=400, target_tokens=300, max_tokens=200)
