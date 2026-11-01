"""Evaluate an ingestion manifest against independently loaded Zotero records."""

import json
from pathlib import Path

from related_work_assistant.evaluation.metadata import compare_metadata, summarize
from related_work_assistant.evaluation.zotero import attachment_index, read_zotero
from related_work_assistant.extraction.schemas import PaperMetadata
from related_work_assistant.pipelines.ingestion import write_json


def evaluate_manifest(
    manifest: Path, rdf: Path, storage_root: Path, raw_pdfs_dir: Path, output: Path
) -> dict:
    references = read_zotero(rdf)
    index = attachment_index(references)
    entries = json.loads(manifest.read_text())
    rows = []
    seen = set()
    for entry in entries:
        row = {"path": entry["path"], "sha256": entry.get("sha256")}
        if entry["status"] not in ("succeeded", "skipped"):
            rows.append({**row, "status": "extraction_failed"})
            continue
        relative = (storage_root / entry["path"]).resolve().relative_to(raw_pdfs_dir).as_posix()
        matches = index.get(relative, [])
        if not matches:
            rows.append({**row, "status": "unmatched"})
            continue
        if len({m.metadata.model_dump_json() for m in matches}) > 1:
            rows.append(
                {
                    **row,
                    "status": "ambiguous_reference",
                    "reference_ids": [m.item_id for m in matches],
                }
            )
            continue
        if entry["sha256"] in seen:
            rows.append({**row, "status": "duplicate_pdf_excluded"})
            continue
        seen.add(entry["sha256"])
        prediction = PaperMetadata.model_validate_json(
            (storage_root / entry["artifact_dir"] / "metadata.json").read_text()
        )
        ref = matches[0]
        rows.append(
            {
                **row,
                "status": "evaluated",
                "reference_id": ref.item_id,
                "prediction": prediction.model_dump(),
                "reference": ref.metadata.model_dump(),
                "zotero_tags_not_scored_as_keywords": ref.tags,
                "scores": compare_metadata(prediction, ref.metadata),
            }
        )
    report = {
        "manifest": str(manifest.relative_to(storage_root)),
        "reference_file": str(rdf.relative_to(storage_root)),
        "reference_records": len(references),
        "input_files": len(entries),
        "evaluated_unique_pdfs": sum(r["status"] == "evaluated" for r in rows),
        "summary": summarize(rows),
        "rows": rows,
        "notes": [
            "Zotero is a reference, not guaranteed ground truth.",
            "Matching uses attachment paths, never extracted metadata.",
            "Exact PDF duplicates excluded from aggregate scores.",
            "Missing reference fields excluded from denominators; missing predictions score zero.",
            "Zotero tags are not publisher keywords; keyword scores unavailable.",
            "Token F1 ignores token order; author exact match retains author order.",
        ],
    }
    write_json(output, report)
    lines = [
        "# Metadata evaluation",
        "",
        f"Evaluated unique PDFs: {report['evaluated_unique_pdfs']} / {len(entries)} input paths.",
        "",
        "| Field | References | Extracted | Exact match | Token F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, stats in report["summary"].items():
        numbers = [
            "n/a" if stats[k] is None else f"{stats[k]:.3f}"
            for k in ("extraction_coverage", "exact_match_rate", "mean_token_f1")
        ]
        lines.append(f"| {name} | {stats['reference_count']} | " + " | ".join(numbers) + " |")
    lines += [
        "",
        *[f"- {note}" for note in report["notes"]],
        "",
        "Per-file predictions, references, unmatched records and differences are in the JSON report.",
    ]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return report
