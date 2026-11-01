"""Read Zotero RDF as reference data and link by attachment path, not predicted title."""

import re
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlparse

from defusedxml import ElementTree as ET
from pydantic import BaseModel, Field
from rdflib import RDF, Graph, Literal, Namespace

from related_work_assistant.extraction.schemas import Author, PaperMetadata

DC = Namespace("http://purl.org/dc/elements/1.1/")
DCT = Namespace("http://purl.org/dc/terms/")
BIB = Namespace("http://purl.org/net/biblio#")
Z = Namespace("http://www.zotero.org/namespaces/export#")
FOAF = Namespace("http://xmlns.com/foaf/0.1/")
LINK = Namespace("http://purl.org/rss/1.0/modules/link/")


class Reference(BaseModel):
    item_id: str
    metadata: PaperMetadata
    attachment_paths: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


def read_zotero(path: Path) -> list[Reference]:
    # Validate XML before RDF parsing; encode Zotero's literal spaces in resource URIs.
    root = ET.fromstring(path.read_bytes())
    for element in root.iter():
        for key in ("resource", "about"):
            attribute = "{" + str(RDF) + "}" + key
            if attribute in element.attrib:
                element.set(attribute, quote(element.get(attribute), safe="/:#%?=&+@"))
    graph = Graph().parse(data=ET.tostring(root), format="xml", publicID=path.resolve().as_uri())

    def value(subject, predicate):
        result = graph.value(subject, predicate)
        return str(result).strip() if result is not None else None

    references = []
    for subject in sorted(set(graph.subjects(Z.itemType, None)), key=str):
        if value(subject, Z.itemType) in ("attachment", "note"):
            continue
        paths = []
        for attachment in graph.objects(subject, LINK.link):
            for resource in graph.objects(attachment, Z.path):
                parts = PurePosixPath(unquote(urlparse(str(resource)).path)).parts
                if "files" in parts:
                    relative = PurePosixPath(*parts[parts.index("files") + 1 :])
                    if ".." not in relative.parts:
                        paths.append(relative.as_posix())
        authors = []
        seq = graph.value(subject, BIB.authors)
        if seq is not None:
            ordered = sorted(
                (
                    (int(str(p).rsplit("_", 1)[1]), o)
                    for p, o in graph.predicate_objects(seq)
                    if str(p).startswith(str(RDF) + "_")
                ),
                key=lambda pair: pair[0],
            )
            for _, person in ordered:
                first = value(person, FOAF.givenName)
                last = value(person, FOAF.surname) or value(person, FOAF.name)
                if first or last:
                    authors.append(Author(first_name=first, last_name=last))
        containers = list(graph.objects(subject, DCT.isPartOf))
        venue = next((value(c, DC.title) for c in containers if value(c, DC.title)), None)
        identifiers = []
        # Zotero sometimes places this article's DOI on the nested journal node.
        for entity in [subject, *containers]:
            for identifier in graph.objects(entity, DC.identifier):
                identifiers.append(
                    str(identifier)
                    if isinstance(identifier, Literal)
                    else value(identifier, RDF.value) or ""
                )
        doi = None
        for identifier in identifiers:
            match = re.search(r"10\.\d{4,9}/\S+", identifier, re.I)
            if match:
                doi = match[0]
                break
        date = value(subject, DC.date) or ""
        year = re.search(r"\b(18\d{2}|19\d{2}|20\d{2}|2100)\b", date)
        tags = sorted(str(t) for t in graph.objects(subject, DC.subject) if isinstance(t, Literal))
        # Zotero subjects are user tags, not verified publisher keywords.
        references.append(
            Reference(
                item_id=str(subject),
                attachment_paths=sorted(set(paths)),
                tags=tags,
                metadata=PaperMetadata(
                    title=value(subject, DC.title),
                    authors=authors,
                    year=int(year[0]) if year else None,
                    venue=venue,
                    doi=doi,
                    abstract=value(subject, DCT.abstract),
                ),
            )
        )
    return references


def attachment_index(references: list[Reference]) -> dict[str, list[Reference]]:
    index: dict[str, list[Reference]] = {}
    for ref in references:
        for path in ref.attachment_paths:
            index.setdefault(path, []).append(ref)
    return index
