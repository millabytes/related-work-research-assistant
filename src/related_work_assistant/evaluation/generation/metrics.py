"""Pure ROUGE, citation, claim and evidence-linking metrics."""

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

BENCHMARK_CITATION_RE = re.compile(r"@cite_\d+|\[R\d+\]", re.IGNORECASE)
CONCRETE_CITATION_RE = re.compile(r"\[([A-Za-z0-9_.:-]+)\]")
TOKEN_RE = re.compile(r"<cite>|[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")


def normalize_benchmark_citations(text: str) -> str:
    return BENCHMARK_CITATION_RE.sub(" <cite> ", text)


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_benchmark_citations(text).casefold())


def _f1(overlap: int, prediction_count: int, reference_count: int) -> float:
    if prediction_count == 0 or reference_count == 0 or overlap == 0:
        return 0.0
    precision = overlap / prediction_count
    recall = overlap / reference_count
    return 2 * precision * recall / (precision + recall)


def _ngrams(tokens: list[str], size: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1))


def _lcs_length(first: list[str], second: list[str]) -> int:
    if len(first) < len(second):
        first, second = second, first
    previous = [0] * (len(second) + 1)
    for left in first:
        current = [0]
        for index, right in enumerate(second, 1):
            current.append(
                previous[index - 1] + 1
                if left == right
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


def rouge_scores(prediction: str, reference: str) -> dict[str, float]:
    predicted = _tokens(prediction)
    expected = _tokens(reference)
    scores: dict[str, float] = {}
    for size, name in ((1, "rouge1"), (2, "rouge2")):
        predicted_ngrams = _ngrams(predicted, size)
        expected_ngrams = _ngrams(expected, size)
        overlap = sum((predicted_ngrams & expected_ngrams).values())
        scores[name] = _f1(overlap, sum(predicted_ngrams.values()), sum(expected_ngrams.values()))
    scores["rougeL"] = _f1(_lcs_length(predicted, expected), len(predicted), len(expected))
    return scores


def concrete_citations(text: str) -> list[str]:
    return CONCRETE_CITATION_RE.findall(text)


def citation_metrics(
    text: str, allowed: Iterable[str], *, unresolved: bool = False
) -> dict[str, Any]:
    if unresolved:
        return {
            "status": "unresolved",
            "validity": None,
            "coverage": None,
            "concrete_citation_count": None,
            "valid_citations": [],
            "invalid_citations": [],
        }
    allowed_set = set(allowed)
    found = concrete_citations(text)
    valid = [citation for citation in found if citation in allowed_set]
    invalid = [citation for citation in found if citation not in allowed_set]
    return {
        "status": "evaluated",
        "validity": len(valid) / len(found) if found else 0.0,
        "coverage": len(set(valid)) / len(allowed_set) if allowed_set else None,
        "concrete_citation_count": len(found),
        "valid_citations": list(dict.fromkeys(valid)),
        "invalid_citations": list(dict.fromkeys(invalid)),
    }


def split_cited_claims(text: str) -> list[dict[str, Any]]:
    claims = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text.strip()):
        sentence = sentence.strip()
        citations = concrete_citations(sentence)
        if sentence and citations:
            claims.append(
                {
                    "claim_index": len(claims),
                    "text": sentence,
                    "nli_hypothesis": re.sub(r"\s+", " ", CONCRETE_CITATION_RE.sub("", sentence)).strip(),
                    "citations": list(dict.fromkeys(citations)),
                }
            )
    return claims


def link_claims_to_evidence(text: str, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Link each citation only to chunks belonging to its explicitly cited paper."""
    recommendations = {
        str(paper["citation_key"]): str(paper["paper_id"])
        for paper in snapshot.get("recommended_papers", [])
    }
    chunks = {str(chunk["chunk_id"]): chunk for chunk in snapshot.get("chunk_catalog", [])}
    evidence_by_paper: dict[str, list[dict[str, Any]]] = {}
    for pair in snapshot.get("chunk_pairs", []):
        paper_id = str(pair["recommended_paper_id"])
        chunk = chunks.get(str(pair["recommended_chunk_id"]))
        if chunk and chunk not in evidence_by_paper.setdefault(paper_id, []):
            evidence_by_paper[paper_id].append(chunk)

    links = []
    for claim in split_cited_claims(text):
        for citation in claim["citations"]:
            paper_id = recommendations.get(citation)
            if paper_id is None:
                links.append(
                    {
                        **claim,
                        "citation": citation,
                        "source_paper_id": None,
                        "status": "invalid_citation",
                        "evidence": [],
                    }
                )
                continue
            evidence = evidence_by_paper.get(paper_id, [])
            links.append(
                {
                    **claim,
                    "citation": citation,
                    "source_paper_id": paper_id,
                    "status": "linked" if evidence else "no_source_evidence",
                    "evidence": [
                        {
                            "chunk_id": str(chunk["chunk_id"]),
                            "text": str(chunk["text"]),
                            "section_heading": chunk.get("section_heading"),
                            "page_start": chunk.get("page_start"),
                            "page_end": chunk.get("page_end"),
                        }
                        for chunk in evidence
                    ],
                }
            )
    return links


def aggregate_numeric(rows: Iterable[dict[str, Any]], keys: Iterable[str]) -> dict[str, float | None]:
    rows = list(rows)
    result: dict[str, float | None] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        result[key] = sum(values) / len(values) if values else None
    return result
