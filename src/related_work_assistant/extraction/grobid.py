"""Local GROBID header and structured full-text extraction."""

import re
import time
from pathlib import Path

import httpx
from defusedxml import ElementTree as ET

from related_work_assistant.extraction.schemas import Author, PaperMetadata

NS = {"t": "http://www.tei-c.org/ns/1.0"}


class GrobidError(RuntimeError):
    pass


def node_text(node) -> str | None:
    if node is None:
        return None
    # Block elements need separators even when TEI is serialized without whitespace.
    parts = []

    def collect(element):
        if element.text:
            parts.append(element.text)
        for child in element:
            collect(child)
            if child.tag.rsplit("}", 1)[-1] in {"p", "div", "head", "ab"}:
                parts.append(" ")
            if child.tail:
                parts.append(child.tail)

    collect(node)
    return " ".join("".join(parts).split()) or None


def parse_header(xml: str) -> PaperMetadata:
    try:
        root = ET.fromstring(xml)
    except Exception as exc:
        raise GrobidError("GROBID returned invalid or unsafe XML") from exc
    header = root.find("t:teiHeader", NS)
    if header is None:
        raise GrobidError("GROBID response has no TEI header")
    bibl = header.find("t:fileDesc/t:sourceDesc/t:biblStruct", NS)
    if bibl is None:
        raise GrobidError("GROBID response has no bibliographic record")
    title = node_text(bibl.find("t:analytic/t:title[@level='a']", NS))
    if not title:
        title = node_text(header.find("t:fileDesc/t:titleStmt/t:title", NS))
    authors = []
    for person in bibl.findall("t:analytic/t:author/t:persName", NS):
        first = " ".join(filter(None, (node_text(n) for n in person.findall("t:forename", NS))))
        last = " ".join(filter(None, (node_text(n) for n in person.findall("t:surname", NS))))
        if first or last:
            authors.append(Author(first_name=first or None, last_name=last or None))
    venue = None
    for level in ("j", "m"):
        venue = node_text(bibl.find(f"t:monogr/t:title[@level='{level}']", NS))
        if venue:
            break
    year = None
    for date in bibl.findall("t:monogr/t:imprint/t:date", NS):
        match = re.search(
            r"\b(18\d{2}|19\d{2}|20\d{2}|2100)\b", date.get("when", "") or node_text(date) or ""
        )
        if match:
            year = int(match[1])
            break
    doi = next(
        (node_text(n) for n in bibl.findall(".//t:idno", NS) if n.get("type", "").lower() == "doi"),
        None,
    )
    keywords = []
    for keyword in header.findall("t:profileDesc/t:textClass/t:keywords", NS):
        terms = keyword.findall("t:term", NS)
        if terms:
            keywords.extend(filter(None, (node_text(n) for n in terms)))
        elif node_text(keyword):
            keywords.extend(v.strip() for v in re.split(r"[;,]", node_text(keyword)) if v.strip())
    return PaperMetadata(
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        doi=doi,
        abstract=node_text(header.find("t:profileDesc/t:abstract", NS)),
        keywords=list(dict.fromkeys(keywords)),
    )


class GrobidClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8070",
        timeout: float = 180,
        transport: httpx.BaseTransport | None = None,
    ):
        self.client = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            timeout=timeout,
            transport=transport,
            trust_env=False,
            follow_redirects=False,
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.client.close()

    def version(self) -> str:
        try:
            response = self.client.get("api/version")
            response.raise_for_status()
            return response.text.strip().strip('"')
        except httpx.HTTPError as exc:
            raise GrobidError(
                "GROBID unavailable; start docker compose --profile extraction up -d grobid"
            ) from exc

    def extract(self, pdf: Path) -> tuple[PaperMetadata, str]:
        for attempt in range(3):
            try:
                with pdf.open("rb") as stream:
                    response = self.client.post(
                        "api/processHeaderDocument",
                        files={"input": (pdf.name, stream, "application/pdf")},
                        data={"consolidateHeader": "0", "includeRawAffiliations": "1"},
                        headers={"Accept": "application/xml"},
                    )
                if response.status_code == 503 and attempt < 2:
                    time.sleep(attempt + 1)
                    continue
                if response.status_code == 204:
                    raise GrobidError("GROBID could not extract metadata (HTTP 204)")
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise GrobidError("GROBID request failed or timed out") from exc
            return parse_header(response.text), response.text
        raise GrobidError("GROBID is busy")

    def extract_fulltext(self, pdf: Path) -> str:
        """Return lossless TEI with sentence, paragraph, and heading coordinates."""
        for attempt in range(3):
            try:
                with pdf.open("rb") as stream:
                    response = self.client.post(
                        "api/processFulltextDocument",
                        files=[
                            ("input", (pdf.name, stream, "application/pdf")),
                            ("consolidateHeader", (None, "0")),
                            ("consolidateCitations", (None, "0")),
                            ("segmentSentences", (None, "1")),
                            ("generateIDs", (None, "1")),
                            ("teiCoordinates", (None, "s")),
                            ("teiCoordinates", (None, "p")),
                            ("teiCoordinates", (None, "head")),
                        ],
                        headers={"Accept": "application/xml"},
                    )
                if response.status_code == 503 and attempt < 2:
                    time.sleep(5 * (attempt + 1))
                    continue
                if response.status_code == 204:
                    raise GrobidError("GROBID could not extract full text (HTTP 204)")
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise GrobidError("GROBID full-text request failed or timed out") from exc
            return response.text
        raise GrobidError("GROBID is busy")
