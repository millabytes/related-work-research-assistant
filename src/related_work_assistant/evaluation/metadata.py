"""Field-level evaluation with explicit reference coverage and per-file differences."""

import re
import unicodedata
from collections import Counter

from related_work_assistant.extraction.schemas import PaperMetadata


def normalize_text(value: str | None) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold()
    return " ".join(re.findall(r"\w+", value))


def token_f1(prediction: str | None, reference: str | None) -> float:
    p, r = Counter(normalize_text(prediction).split()), Counter(normalize_text(reference).split())
    common = sum((p & r).values())
    return 2 * common / (sum(p.values()) + sum(r.values())) if p or r else 1.0


def compare_metadata(prediction: PaperMetadata, reference: PaperMetadata) -> dict:
    results = {}
    for field in ("title", "year", "venue", "doi", "abstract", "authors", "keywords"):
        actual, expected = getattr(prediction, field), getattr(reference, field)
        if not expected:
            results[field] = {"reference_available": False, "extracted": bool(actual)}
            continue
        if field == "authors":
            p = [normalize_text(a.full_name) for a in actual]
            r = [normalize_text(a.full_name) for a in expected]
            match = p == r
            score = token_f1(" ".join(p), " ".join(r))
        elif field == "keywords":
            p, r = {normalize_text(k) for k in actual}, {normalize_text(k) for k in expected}
            match = p == r
            score = 2 * len(p & r) / (len(p) + len(r))
        elif field == "year":
            match, score = actual == expected, float(actual == expected)
        elif field == "doi":
            match, score = actual == expected, float(actual == expected)
        else:
            match = normalize_text(actual) == normalize_text(expected)
            score = token_f1(actual, expected)
        results[field] = {
            "reference_available": True,
            "extracted": bool(actual),
            "exact_match": match,
            "token_f1": score,
        }
    return results


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for field in ("title", "year", "venue", "doi", "abstract", "authors", "keywords"):
        available = [
            r["scores"][field]
            for r in rows
            if "scores" in r and r["scores"][field]["reference_available"]
        ]
        n = len(available)
        summary[field] = {
            "reference_count": n,
            "extraction_coverage": sum(v["extracted"] for v in available) / n if n else None,
            "exact_match_rate": sum(v["exact_match"] for v in available) / n if n else None,
            "mean_token_f1": sum(v["token_f1"] for v in available) / n if n else None,
        }
    return summary
