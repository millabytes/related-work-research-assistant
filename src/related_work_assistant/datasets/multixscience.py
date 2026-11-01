"""Citation-aware Multi-XScience preparation for future Qwen fine-tuning."""

import gzip
import json
import os
import re
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.request import Request, urlopen

from related_work_assistant.core.config import PROJECT_ROOT

DATASET_REPOSITORY = "https://github.com/yaolu/Multi-XScience"
RAW_BASE_URL = "https://raw.githubusercontent.com/yaolu/Multi-XScience/master/data"
QWEN_REPOSITORY = "Qwen/Qwen2.5-0.5B-Instruct"
PREPROCESSING_VERSION = "multixscience-qwen-citations-v1"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data/datasets/multixscience"
DEFAULT_MODEL_CACHE = PROJECT_ROOT / "models_cache/huggingface"
SPLIT_FILES = {"train": "train.json.gz", "validation": "val.json.gz", "test": "test.json.gz"}
CITATION_RE = re.compile(r"@cite_(\d+)")
MISSING_ABSTRACT_RE = re.compile(r"^cited reference @cite_\d+ has no abstract$")

INVALID_REASON_EXPLANATIONS = {
    "cited_reference_missing_abstract": (
        "The citation marker has a ref_abstract entry, but that referenced paper's abstract "
        "is empty, so the target citation cannot be grounded in the model input."
    ),
    "target_citation_missing_from_references": (
        "The Related Work target contains a citation marker that has no matching ref_abstract "
        "entry."
    ),
    "no_reference_abstracts": (
        "The target contains citations, but the example supplies no usable reference mapping."
    ),
    "missing_target_abstract": "The main paper abstract required for the user input is empty.",
    "missing_related_work_target": "The human-written Related Work training target is empty.",
    "unstable_assistant_boundary": (
        "The tokenizer chat template does not provide a stable assistant-only loss boundary."
    ),
    "token_overflow": "The complete, untruncated chat sequence exceeds the configured limit.",
    "unknown": "The invalid reason is not recognized by this preprocessing version.",
}

SYSTEM_MESSAGE = (
    "You write concise academic Related Work prose grounded only in the supplied "
    "target abstract and reference abstracts."
)
USER_INSTRUCTION = (
    "Write the Related Work text for the target paper. Compare the referenced work with "
    "the target paper, synthesize methods and contributions, and cite references only with "
    "their supplied [R1], [R2], ... identifiers. Return only the academic prose."
)


class ChatTokenizer(Protocol):
    name_or_path: str

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]: ...


@dataclass(frozen=True)
class PreparedExample:
    example_id: str
    split: str
    split_index: int
    aid: str | None
    mid: str | None
    target_abstract: str
    references: list[dict[str, str | None]]
    citation_mapping: dict[str, str]
    original_citation_markers: list[str]
    original_related_work: str
    assistant_target: str
    input_messages: list[dict[str, str]]
    messages: list[dict[str, str]]
    input_token_count: int
    assistant_token_start: int
    assistant_token_count: int
    total_token_count: int
    max_sequence_tokens: int
    loss_scope: str = "assistant_only"
    preprocessing_version: str = PREPROCESSING_VERSION


@dataclass(frozen=True)
class InvalidExample:
    example_id: str
    split: str
    split_index: int
    reasons: list[str]
    original_citation_markers: list[str]


def invalid_reason_code(reason: str) -> str:
    """Collapse marker-specific messages into stable statistical categories."""
    if MISSING_ABSTRACT_RE.fullmatch(reason):
        return "cited_reference_missing_abstract"
    if reason.startswith("target citations missing from references:"):
        return "target_citation_missing_from_references"
    if reason == "target has citations but no reference abstracts":
        return "no_reference_abstracts"
    if reason == "missing target abstract":
        return "missing_target_abstract"
    if reason == "missing related_work target":
        return "missing_related_work_target"
    if reason == "tokenizer chat template does not expose a stable assistant boundary":
        return "unstable_assistant_boundary"
    if reason.startswith("token overflow:"):
        return "token_overflow"
    return "unknown"


def _reason_codes(row: InvalidExample | dict[str, Any]) -> set[str]:
    reasons = row.reasons if isinstance(row, InvalidExample) else row.get("reasons", [])
    return {invalid_reason_code(str(reason)) for reason in reasons}


def summarize_invalid_examples(
    rows: Iterable[InvalidExample | dict[str, Any]], *, sample_limit: int = 3
) -> dict[str, Any]:
    """Count invalid examples (not individual markers) and retain representative samples."""
    counts: dict[str, int] = {}
    candidates: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = asdict(row) if isinstance(row, InvalidExample) else row
        for code in _reason_codes(row):
            counts[code] = counts.get(code, 0) + 1
            candidates.setdefault(code, []).append(
                {
                    "example_id": value.get("example_id"),
                    "split": value.get("split"),
                    "split_index": value.get("split_index"),
                    "original_citation_markers": value.get("original_citation_markers", []),
                    "raw_reasons": value.get("reasons", []),
                    "explanation": INVALID_REASON_EXPLANATIONS[code],
                }
            )
    samples: dict[str, list[dict[str, Any]]] = {}
    for code, values in candidates.items():
        selected: list[dict[str, Any]] = []
        for split in SPLIT_FILES:
            match = next((value for value in values if value["split"] == split), None)
            if match is not None:
                selected.append(match)
            if len(selected) >= sample_limit:
                break
        for value in values:
            if len(selected) >= sample_limit:
                break
            if value not in selected:
                selected.append(value)
        samples[code] = selected
    return {"invalid_by_reason": counts, "invalid_examples_by_reason": samples}


def _read_more(stream: Any, buffer: str, chunk_size: int) -> tuple[str, bool]:
    part = stream.read(chunk_size)
    return buffer + part, not bool(part)


def iter_json_array(path: Path, *, chunk_size: int = 1024 * 1024) -> Iterator[dict[str, Any]]:
    """Stream objects from the official gzipped JSON array without loading it all."""
    decoder = json.JSONDecoder()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        buffer, eof = _read_more(stream, "", chunk_size)
        position = 0
        started = False
        while True:
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if not started:
                    if position >= len(buffer) and not eof:
                        buffer, eof = _read_more(stream, buffer[position:], chunk_size)
                        position = 0
                        continue
                    if position >= len(buffer) or buffer[position] != "[":
                        raise ValueError(f"{path} is not a JSON array")
                    position += 1
                    started = True
                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1
                break

            if position < len(buffer) and buffer[position] == "]":
                return
            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError as exc:
                if eof:
                    raise ValueError(f"Invalid JSON array in {path}: {exc}") from exc
                buffer, eof = _read_more(stream, buffer[position:], chunk_size)
                position = 0
                continue
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object in {path}, got {type(value).__name__}")
            yield value
            position = end
            if position > chunk_size:
                buffer = buffer[position:]
                position = 0


def download_raw_splits(dataset_dir: Path = DEFAULT_DATASET_DIR) -> dict[str, Path]:
    """Download missing official archives; existing non-empty files are never replaced."""
    raw_dir = dataset_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, filename in SPLIT_FILES.items():
        destination = raw_dir / filename
        paths[split] = destination
        if destination.is_file() and destination.stat().st_size > 0:
            continue
        request = Request(f"{RAW_BASE_URL}/{filename}", headers={"User-Agent": "rw-assistant/0.1"})
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            if temporary.stat().st_size == 0:
                raise RuntimeError(f"Downloaded empty file for {split}")
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    return paths


def _reference_items(ref_abstract: object) -> list[tuple[str, dict[str, Any]]]:
    if not isinstance(ref_abstract, dict):
        return []
    items = [
        (str(marker), value)
        for marker, value in ref_abstract.items()
        if CITATION_RE.fullmatch(str(marker)) and isinstance(value, dict)
    ]
    return sorted(items, key=lambda item: int(CITATION_RE.fullmatch(item[0]).group(1)))


def build_user_message(target_abstract: str, references: list[dict[str, str | None]]) -> str:
    blocks = [USER_INSTRUCTION, "", "Target paper abstract:", target_abstract.strip(), "", "References:"]
    for reference in references:
        blocks.extend(
            [
                "",
                f"Reference {reference['citation_id']}:",
                str(reference["abstract"]).strip(),
            ]
        )
    return "\n".join(blocks)


def prepare_example(
    raw: dict[str, Any],
    *,
    split: str,
    split_index: int,
    tokenizer: ChatTokenizer,
    max_sequence_tokens: int = 32768,
) -> PreparedExample | InvalidExample:
    """Map citations, build Qwen chat messages, and reject invalid/overflow examples."""
    aid = str(raw["aid"]) if raw.get("aid") is not None else None
    mid = str(raw["mid"]) if raw.get("mid") is not None else None
    example_id = aid or mid or f"{split}:{split_index}"
    target_abstract = str(raw.get("abstract") or "").strip()
    original_target = str(raw.get("related_work") or "").strip()
    markers = list(dict.fromkeys(CITATION_RE.findall(original_target)))
    original_markers = [f"@cite_{number}" for number in markers]
    reasons: list[str] = []
    if not target_abstract:
        reasons.append("missing target abstract")
    if not original_target:
        reasons.append("missing related_work target")

    references: list[dict[str, str | None]] = []
    citation_mapping: dict[str, str] = {}
    for ordinal, (marker, value) in enumerate(_reference_items(raw.get("ref_abstract")), start=1):
        citation_id = f"[R{ordinal}]"
        abstract = str(value.get("abstract") or "").strip()
        citation_mapping[marker] = citation_id
        references.append(
            {
                "citation_id": citation_id,
                "original_marker": marker,
                "mid": str(value["mid"]) if value.get("mid") is not None else None,
                "abstract": abstract,
            }
        )
        if marker in original_markers and not abstract:
            reasons.append(f"cited reference {marker} has no abstract")

    missing = sorted(set(original_markers) - set(citation_mapping))
    if missing:
        reasons.append(f"target citations missing from references: {', '.join(missing)}")
    if original_markers and not references:
        reasons.append("target has citations but no reference abstracts")
    if reasons:
        return InvalidExample(example_id, split, split_index, reasons, original_markers)

    assistant_target = CITATION_RE.sub(
        lambda match: citation_mapping[f"@cite_{match.group(1)}"], original_target
    )
    input_messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": build_user_message(target_abstract, references)},
    ]
    messages = [*input_messages, {"role": "assistant", "content": assistant_target}]
    prompt_ids = tokenizer.apply_chat_template(
        input_messages, tokenize=True, add_generation_prompt=True
    )
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        return InvalidExample(
            example_id,
            split,
            split_index,
            ["tokenizer chat template does not expose a stable assistant boundary"],
            original_markers,
        )
    if len(full_ids) > max_sequence_tokens:
        return InvalidExample(
            example_id,
            split,
            split_index,
            [f"token overflow: {len(full_ids)} > {max_sequence_tokens}"],
            original_markers,
        )
    return PreparedExample(
        example_id=example_id,
        split=split,
        split_index=split_index,
        aid=aid,
        mid=mid,
        target_abstract=target_abstract,
        references=references,
        citation_mapping=citation_mapping,
        original_citation_markers=original_markers,
        original_related_work=original_target,
        assistant_target=assistant_target,
        input_messages=input_messages,
        messages=messages,
        input_token_count=len(prompt_ids),
        assistant_token_start=len(prompt_ids),
        assistant_token_count=len(full_ids) - len(prompt_ids),
        total_token_count=len(full_ids),
        max_sequence_tokens=max_sequence_tokens,
    )


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
        temporary = Path(file.name)
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
        for row in rows:
            json.dump(asdict(row) if hasattr(row, "__dataclass_fields__") else row, file, ensure_ascii=False)
            file.write("\n")
        temporary = Path(file.name)
    os.replace(temporary, path)


@contextmanager
def _atomic_jsonl_writer(path: Path) -> Iterator[Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as file:
            temporary = Path(file.name)
            yield file
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def load_qwen_tokenizer(
    *, revision: str | None = None, model_cache: Path = DEFAULT_MODEL_CACHE
) -> ChatTokenizer:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        QWEN_REPOSITORY, revision=revision, cache_dir=model_cache
    )


def prepare_dataset(
    *,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    tokenizer: ChatTokenizer,
    tokenizer_revision: str | None = None,
    limit: int | None = None,
    max_sequence_tokens: int = 32768,
    raw_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Prepare all official splits, optionally sampling the first ``limit`` rows each."""
    raw_paths = raw_paths or download_raw_splits(dataset_dir)
    all_invalid: list[InvalidExample] = []
    split_stats: dict[str, dict[str, int]] = {}
    for split in SPLIT_FILES:
        seen = 0
        prepared_count = 0
        invalid: list[InvalidExample] = []
        with _atomic_jsonl_writer(dataset_dir / f"{split}.jsonl") as output:
            for split_index, raw in enumerate(iter_json_array(raw_paths[split])):
                if limit is not None and seen >= limit:
                    break
                seen += 1
                result = prepare_example(
                    raw,
                    split=split,
                    split_index=split_index,
                    tokenizer=tokenizer,
                    max_sequence_tokens=max_sequence_tokens,
                )
                if isinstance(result, PreparedExample):
                    json.dump(asdict(result), output, ensure_ascii=False)
                    output.write("\n")
                    prepared_count += 1
                else:
                    invalid.append(result)
        all_invalid.extend(invalid)
        invalid_summary = summarize_invalid_examples(invalid)
        split_stats[split] = {
            "seen": seen,
            "prepared": prepared_count,
            "invalid": len(invalid),
            "invalid_by_reason": invalid_summary["invalid_by_reason"],
            "token_overflow": sum(
                any(reason.startswith("token overflow:") for reason in row.reasons)
                for row in invalid
            ),
        }

    _write_jsonl(dataset_dir / "invalid_examples.jsonl", all_invalid)
    config = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "source_repository": DATASET_REPOSITORY,
        "raw_files": SPLIT_FILES,
        "model_id": QWEN_REPOSITORY,
        "tokenizer_revision": tokenizer_revision or "main",
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", QWEN_REPOSITORY),
        "max_sequence_tokens": max_sequence_tokens,
        "limit_per_split": limit,
        "citation_policy": "@cite_N mapped bijectively to [R1]...[Rn] per example",
        "loss_scope": "assistant_only",
        "silent_truncation": False,
    }
    stats = {
        "created_at": datetime.now(UTC).isoformat(),
        "sampled": limit is not None,
        "splits": split_stats,
        "total_prepared": sum(row["prepared"] for row in split_stats.values()),
        "total_invalid": sum(row["invalid"] for row in split_stats.values()),
    }
    stats.update(summarize_invalid_examples(all_invalid))
    stats["invalid_reason_explanations"] = INVALID_REASON_EXPLANATIONS
    _atomic_json(dataset_dir / "preprocessing_config.json", config)
    _atomic_json(dataset_dir / "statistics.json", stats)
    return {"config": config, "statistics": stats, "dataset_dir": str(dataset_dir)}


def inspect_examples(
    *, dataset_dir: Path = DEFAULT_DATASET_DIR, split: str = "train", limit: int = 1
) -> list[dict[str, Any]]:
    if split not in SPLIT_FILES:
        raise ValueError(f"split must be one of: {', '.join(SPLIT_FILES)}")
    path = dataset_dir / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Prepared split not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if len(rows) >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _empty_recovery_counts() -> dict[str, int]:
    return {
        "invalid_examples_only_missing_reference_abstract": 0,
        "invalid_examples_with_missing_reference_mapping": 0,
        "missing_cited_reference_occurrences": 0,
        "missing_occurrences_with_mid": 0,
        "missing_occurrences_recoverable_by_exact_mid": 0,
        "invalid_examples_with_at_least_one_grounded_citation": 0,
        "invalid_examples_with_no_grounded_citations": 0,
        "grounded_citation_occurrences": 0,
        "examples_all_missing_abstracts_recoverable_by_exact_mid": 0,
        "examples_partially_recoverable_by_exact_mid": 0,
        "examples_not_recoverable_by_exact_mid": 0,
    }


def analyze_invalid_dataset(dataset_dir: Path = DEFAULT_DATASET_DIR) -> dict[str, Any]:
    """Audit an existing full run without tokenizing or changing filtering decisions."""
    invalid_path = dataset_dir / "invalid_examples.jsonl"
    statistics_path = dataset_dir / "statistics.json"
    if not invalid_path.is_file() or not statistics_path.is_file():
        raise FileNotFoundError("Run prepare-multixscience before auditing invalid examples")
    raw_paths = {split: dataset_dir / "raw" / filename for split, filename in SPLIT_FILES.items()}
    missing_raw = [str(path) for path in raw_paths.values() if not path.is_file()]
    if missing_raw:
        raise FileNotFoundError(f"Missing cached raw files: {', '.join(missing_raw)}")

    invalid_rows = _load_jsonl(invalid_path)
    summary = summarize_invalid_examples(invalid_rows)
    invalid_indexes: dict[str, set[int]] = {split: set() for split in SPLIT_FILES}
    for row in invalid_rows:
        split = str(row["split"])
        if split in invalid_indexes:
            invalid_indexes[split].add(int(row["split_index"]))

    # A missing reference abstract may be safely recoverable if the exact paper mid has a
    # non-empty abstract elsewhere in the unmodified official dataset.
    mids_with_abstract: set[str] = set()
    for split, path in raw_paths.items():
        for raw in iter_json_array(path):
            if raw.get("mid") is not None and str(raw.get("abstract") or "").strip():
                mids_with_abstract.add(str(raw["mid"]))
            for _marker, value in _reference_items(raw.get("ref_abstract")):
                if value.get("mid") is not None and str(value.get("abstract") or "").strip():
                    mids_with_abstract.add(str(value["mid"]))

    total_audit = _empty_recovery_counts()
    split_audit = {split: _empty_recovery_counts() for split in SPLIT_FILES}
    invalid_by_index = {
        (str(row["split"]), int(row["split_index"])): row for row in invalid_rows
    }
    for split, path in raw_paths.items():
        for split_index, raw in enumerate(iter_json_array(path)):
            if split_index not in invalid_indexes[split]:
                continue
            row = invalid_by_index[(split, split_index)]
            codes = _reason_codes(row)
            counters = split_audit[split]
            if codes == {"cited_reference_missing_abstract"}:
                counters["invalid_examples_only_missing_reference_abstract"] += 1
            if "target_citation_missing_from_references" in codes:
                counters["invalid_examples_with_missing_reference_mapping"] += 1

            ref_values = {
                marker: value for marker, value in _reference_items(raw.get("ref_abstract"))
            }
            cited_markers = {f"@cite_{number}" for number in CITATION_RE.findall(str(raw.get("related_work") or ""))}
            grounded_markers = [
                marker
                for marker in cited_markers
                if marker in ref_values and str(ref_values[marker].get("abstract") or "").strip()
            ]
            counters["grounded_citation_occurrences"] += len(grounded_markers)
            if grounded_markers:
                counters["invalid_examples_with_at_least_one_grounded_citation"] += 1
            else:
                counters["invalid_examples_with_no_grounded_citations"] += 1
            missing_values = [
                ref_values[marker]
                for marker in cited_markers
                if marker in ref_values and not str(ref_values[marker].get("abstract") or "").strip()
            ]
            recoverable: list[bool] = []
            for value in missing_values:
                counters["missing_cited_reference_occurrences"] += 1
                mid = str(value["mid"]) if value.get("mid") is not None else None
                if mid:
                    counters["missing_occurrences_with_mid"] += 1
                found = bool(mid and mid in mids_with_abstract)
                recoverable.append(found)
                if found:
                    counters["missing_occurrences_recoverable_by_exact_mid"] += 1
            if recoverable and all(recoverable):
                counters["examples_all_missing_abstracts_recoverable_by_exact_mid"] += 1
            elif any(recoverable):
                counters["examples_partially_recoverable_by_exact_mid"] += 1
            elif recoverable:
                counters["examples_not_recoverable_by_exact_mid"] += 1

    for counters in split_audit.values():
        for key, value in counters.items():
            total_audit[key] += value

    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    statistics.update(summary)
    statistics["invalid_reason_explanations"] = INVALID_REASON_EXPLANATIONS
    statistics["citation_recovery_audit"] = {
        "method": (
            "Read-only lookup by exact reference mid across target and reference abstracts in "
            "all cached official splits; no examples or abstracts were modified."
        ),
        "total": total_audit,
        "splits": split_audit,
    }
    for split in SPLIT_FILES:
        split_rows = [row for row in invalid_rows if row.get("split") == split]
        statistics["splits"][split]["invalid_by_reason"] = summarize_invalid_examples(
            split_rows
        )["invalid_by_reason"]
    _atomic_json(statistics_path, statistics)
    return statistics
