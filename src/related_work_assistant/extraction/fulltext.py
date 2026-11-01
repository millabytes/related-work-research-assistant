"""Parse GROBID full-text TEI and align its logical text to PyMuPDF geometry."""

import re
from difflib import SequenceMatcher

from defusedxml import ElementTree as ET
from pydantic import BaseModel, Field

from related_work_assistant.extraction.grobid import NS, GrobidError, node_text
from related_work_assistant.extraction.schemas import PDFText, Word

XML_ID = "{http://www.w3.org/XML/1998/namespace}id"


class Coordinate(BaseModel):
    page: int
    x: float
    y: float
    width: float
    height: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.x + self.width, self.y + self.height


class StructuredSentence(BaseModel):
    sentence_id: str | None = None
    text: str
    coordinates: list[Coordinate] = Field(default_factory=list)


class StructuredParagraph(BaseModel):
    paragraph_id: str | None = None
    sentences: list[StructuredSentence]

    @property
    def text(self) -> str:
        return " ".join(sentence.text for sentence in self.sentences)


class StructuredSection(BaseModel):
    heading: str | None = None
    section_type: str
    paragraphs: list[StructuredParagraph]


class StructuredDocument(BaseModel):
    sections: list[StructuredSection]
    page_sizes: dict[int, tuple[float, float]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def parse_coordinates(value: str | None) -> list[Coordinate]:
    result = []
    for raw in (value or "").split(";"):
        parts = raw.split(",")
        if len(parts) != 5:
            continue
        try:
            page = int(parts[0])
            x, y, width, height = (float(part) for part in parts[1:])
        except ValueError:
            continue
        if page >= 1 and width > 0 and height > 0:
            result.append(Coordinate(page=page, x=x, y=y, width=width, height=height))
    return result


def classify_section(heading: str | None, tei_type: str | None = None) -> str:
    value = " ".join(filter(None, (tei_type, heading))).casefold()
    patterns = (
        ("references", r"\b(reference|references|bibliography)\b"),
        ("related_work", r"\b(related work|background|literature review)\b"),
        ("introduction", r"\b(introduction|motivation)\b"),
        ("methods", r"\b(methods?|methodology|approach|model|framework)\b"),
        ("experiments", r"\b(experiments?|experimental|evaluation|setup|implementation)\b"),
        ("results", r"\b(result|results|findings|analysis)\b"),
        ("discussion", r"\b(discussion|limitations?)\b"),
        ("conclusion", r"\b(conclusion|conclusions|future work)\b"),
    )
    return next((kind for kind, pattern in patterns if re.search(pattern, value)), "other")


def _paragraph(element) -> StructuredParagraph | None:
    def technical_noise(value: str) -> bool:
        tokens = re.findall(r"\w+(?:[’'-]\w+)*|[^\w\s]", value)
        if not tokens:
            return True
        numeric = sum(bool(re.fullmatch(r"[\d.]+|[=<>±%]+", token)) for token in tokens)
        alphabetic = sum(any(character.isalpha() for character in token) for token in tokens)
        if len(tokens) <= 3 and alphabetic == 0:
            return True
        return len(tokens) >= 100 and (
            numeric / len(tokens) >= 0.4 or alphabetic / len(tokens) < 0.35
        )

    sentences = []
    sentence_nodes = element.findall(".//t:s", NS)
    if sentence_nodes:
        for sentence in sentence_nodes:
            text = node_text(sentence)
            if text and not technical_noise(text):
                sentences.append(
                    StructuredSentence(
                        sentence_id=sentence.get(XML_ID),
                        text=text,
                        coordinates=parse_coordinates(sentence.get("coords")),
                    )
                )
    else:
        text = node_text(element)
        if text and not technical_noise(text):
            sentences.append(
                StructuredSentence(
                    sentence_id=element.get(XML_ID),
                    text=text,
                    coordinates=parse_coordinates(element.get("coords")),
                )
            )
    if not sentences:
        return None
    return StructuredParagraph(paragraph_id=element.get(XML_ID), sentences=sentences)


def _direct_paragraphs(container) -> list[StructuredParagraph]:
    paragraphs = []
    for child in container:
        name = child.tag.rsplit("}", 1)[-1]
        if name == "p":
            paragraph = _paragraph(child)
            if paragraph:
                paragraphs.append(paragraph)
        elif name in {"list", "quote"}:
            for candidate in child.findall(".//t:p", NS):
                paragraph = _paragraph(candidate)
                if paragraph:
                    paragraphs.append(paragraph)
    return paragraphs


def parse_fulltext(xml: str) -> StructuredDocument:
    try:
        root = ET.fromstring(xml)
    except Exception as exc:
        raise GrobidError("GROBID returned invalid or unsafe full-text XML") from exc
    if root.tag.rsplit("}", 1)[-1] != "TEI":
        raise GrobidError("GROBID full-text response has no TEI root")

    page_sizes = {}
    for position, surface in enumerate(root.findall("t:facsimile/t:surface", NS), start=1):
        try:
            page = int(surface.get("n", position))
            width = float(surface.get("lrx", "0")) - float(surface.get("ulx", "0"))
            height = float(surface.get("lry", "0")) - float(surface.get("uly", "0"))
        except ValueError:
            continue
        if width > 0 and height > 0:
            page_sizes[page] = (width, height)

    sections = []
    abstract = root.find("t:teiHeader/t:profileDesc/t:abstract", NS)
    if abstract is not None:
        paragraphs = [_paragraph(p) for p in abstract.findall(".//t:p", NS)]
        paragraphs = [p for p in paragraphs if p]
        if paragraphs:
            sections.append(
                StructuredSection(
                    heading="Abstract", section_type="abstract", paragraphs=paragraphs
                )
            )

    body = root.find("t:text/t:body", NS)
    if body is None:
        raise GrobidError("GROBID full-text response has no body")

    unheaded = _direct_paragraphs(body)
    if unheaded:
        sections.append(StructuredSection(heading=None, section_type="other", paragraphs=unheaded))

    def visit(div, inherited_heading: str | None = None) -> None:
        heading = node_text(div.find("t:head", NS)) or inherited_heading
        section_type = classify_section(heading, div.get("type"))
        paragraphs = _direct_paragraphs(div)
        if paragraphs and section_type != "references":
            sections.append(
                StructuredSection(heading=heading, section_type=section_type, paragraphs=paragraphs)
            )
        for nested in div.findall("t:div", NS):
            visit(nested, heading)

    for div in body.findall("t:div", NS):
        visit(div)

    if not sections:
        raise GrobidError("GROBID full-text response contains no usable paragraphs")
    return StructuredDocument(sections=sections, page_sizes=page_sizes)


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.casefold()))


def _overlap(word: Word, bbox: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = word.bbox
    bx0, by0, bx1, by1 = bbox
    intersection = max(0.0, min(x1, bx1) - max(x0, bx0)) * max(0.0, min(y1, by1) - max(y0, by0))
    area = max((x1 - x0) * (y1 - y0), 1e-6)
    center_inside = bx0 <= (x0 + x1) / 2 <= bx1 and by0 <= (y0 + y1) / 2 <= by1
    return center_inside or intersection / area >= 0.3


def map_sentence(
    sentence: StructuredSentence,
    source: PDFText,
    page_sizes: dict[int, tuple[float, float]],
) -> dict:
    mapped_regions = []
    matched_words: dict[tuple[int, int, int, int], Word] = {}
    for coordinate in sentence.coordinates:
        if coordinate.page > len(source.pages):
            continue
        page = source.pages[coordinate.page - 1]
        grobid_size = page_sizes.get(coordinate.page, (page.width, page.height))
        scale_x = page.width / grobid_size[0] if grobid_size[0] else 1.0
        scale_y = page.height / grobid_size[1] if grobid_size[1] else 1.0
        x0, y0, x1, y1 = coordinate.bbox
        bbox = (x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y)
        region_words = [word for word in page.words if _overlap(word, bbox)]
        for word in region_words:
            matched_words[(coordinate.page, word.block_index, word.line_index, word.word_index)] = (
                word
            )
        mapped_regions.append(
            {
                "page": coordinate.page,
                "grobid_bbox": list(coordinate.bbox),
                "pymupdf_bbox": list(bbox),
                "coordinate_system": source.coordinate_system,
            }
        )

    ordered = sorted(matched_words.items(), key=lambda item: item[0])
    matched_text = " ".join(word.text for _, word in ordered)
    similarity = SequenceMatcher(None, _normalize(sentence.text), _normalize(matched_text)).ratio()
    if not ordered:
        status = "unmapped"
    elif similarity >= 0.8:
        status = "mapped"
    else:
        status = "weak"
    return {
        "grobid_sentence_id": sentence.sentence_id,
        "grobid_text": sentence.text,
        "grobid_coordinates": [coordinate.model_dump() for coordinate in sentence.coordinates],
        "pymupdf_regions": mapped_regions,
        "pymupdf_word_refs": [
            {
                "page": key[0],
                "block_index": key[1],
                "line_index": key[2],
                "word_index": key[3],
                "text": word.text,
                "bbox": list(word.bbox),
            }
            for key, word in ordered
        ],
        "pymupdf_text": matched_text,
        "text_similarity": round(similarity, 6),
        "mapping_status": status,
    }
