"""Local Transformers generators with strict pre-inference token validation."""

import json
import re
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from related_work_assistant.embeddings.encoders import (
    MAX_MODEL_PARAMETERS,
    Device,
    choose_device,
    resolve_revision,
)

PRIMERA_REPOSITORY = "allenai/PRIMERA-multixscience"
QWEN_REPOSITORY = "Qwen/Qwen2.5-0.5B-Instruct"
PRIMERA_PROMPT_VERSION = "multixscience-source-v2"
PRIMERA_GENERATION_VERSION = "primera-multixscience-v2"
QWEN_PROMPT_VERSION = "qwen-related-work-v5-short-citations"
QWEN_GENERATION_VERSION = "qwen-single-pass-v5"
QWEN_LORA_GENERATION_VERSION = "qwen-lora-single-pass-v1"

UUID_PATTERN = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
INTERNAL_MARKER_PATTERN = re.compile(
    r"(?im)^\s*(?:chunk|section|target paper|related paper|target passage|related passage|comparison\s+\d+|paragraph\s+\d+)\b|^\s*(?:methods|similarities|differences|contributions(?:\s+or\s+findings)?|open issues)\s*:"
)
NON_PROSE_FORMAT_PATTERN = re.compile(r"(?m)^\s*(?:#{1,6}\s|\d+[.)]\s|[-*]\s)")


@dataclass(frozen=True)
class GenerationOutput:
    status: str
    text: str | None
    input_tokens: int
    output_tokens: int | None
    context_limit: int
    input_hash: str
    citations: list[str]
    error: str | None = None
    finish_reason: str | None = None
    validation_errors: tuple[str, ...] = ()


def extract_citations(text: str, allowed: set[str]) -> list[str]:
    found = list(dict.fromkeys(re.findall(r"\[([A-Za-z0-9_.:-]+)\]", text)))
    unknown = sorted(set(found) - allowed)
    if unknown:
        raise ValueError(f"Generated unknown citation keys: {', '.join(unknown)}")
    return found


def validate_budget(input_tokens: int, max_new_tokens: int, safety: int, limit: int) -> str | None:
    if input_tokens + max_new_tokens + safety > limit:
        return (
            f"context_overflow: input={input_tokens}, output_budget={max_new_tokens}, "
            f"safety_margin={safety}, limit={limit}"
        )
    return None


def build_primera_input(snapshot: dict[str, object]) -> str:
    papers = [snapshot["query_paper"], *snapshot["recommended_papers"]]
    # The original PRIMERA Multi-XScience loader places the target-paper abstract
    # first, then every reference abstract, and appends <doc-sep> after each doc.
    return " <doc-sep> ".join(str(paper["abstract"]) for paper in papers) + " <doc-sep>"


def build_qwen_prompt(snapshot: dict[str, object]) -> str:
    query = snapshot["query_paper"]
    recommendations = snapshot["recommended_papers"]
    chunks = {chunk["chunk_id"]: chunk for chunk in snapshot["chunk_catalog"]}
    pairs_by_paper: dict[str, list[dict[str, object]]] = {
        str(paper["paper_id"]): [] for paper in recommendations
    }
    for pair in snapshot["chunk_pairs"]:
        pairs_by_paper[str(pair["recommended_paper_id"])].append(pair)
    lines = [
        "Write 1 or 2 connected academic Related Work paragraphs in English (150-220 words).",
        "Compare concrete methods, contributions, similarities, and differences using the evidence below.",
        "Synthesize and paraphrase; do not copy abstracts, enumerate papers, repeat claims, or add a heading.",
        "Cite only relevant sources with [R1] to [R5]. You do not need to cite every source.",
        "Return only prose paragraphs, with no bullets, labels, preface, or commentary.",
        "\nTarget paper",
        f"Title: {query['title']}",
        f"Abstract: {query['abstract']}",
        "\nRelated sources and paired evidence",
    ]
    for paper_index, paper in enumerate(recommendations, 1):
        lines.extend(
            [
                f"\n[R{paper_index}]",
                f"Title: {paper['title']}",
                f"Abstract: {paper['abstract']}",
            ]
        )
        paper_pairs = sorted(
            pairs_by_paper[str(paper["paper_id"])], key=lambda pair: int(pair["rank"])
        )
        if len(paper_pairs) != 3:
            raise ValueError(f"Expected 3 evidence pairs for related paper {paper_index}")
        for pair_index, pair in enumerate(paper_pairs, 1):
            query_chunk = chunks[str(pair["query_chunk_id"])]
            related_chunk = chunks[str(pair["recommended_chunk_id"])]
            lines.extend(
                [
                    f"Paired evidence {pair_index}",
                    "Target excerpt:",
                    str(query_chunk["text"]),
                    "Related excerpt:",
                    str(related_chunk["text"]),
                ]
            )
    lines.extend(
        [
            "\nWrite the final 1-2 paragraph synthesis now. Use [R1]-[R5] only when the source supports the claim.",
        ]
    )
    return "\n".join(lines)


def qwen_citation_map(snapshot: dict[str, object]) -> dict[str, str]:
    recommendations = snapshot["recommended_papers"]
    if len(recommendations) != 5:
        raise ValueError("Qwen citation mapping requires exactly 5 recommended papers")
    return {
        f"R{index}": str(paper["citation_key"])
        for index, paper in enumerate(recommendations, 1)
    }


def build_qwen_messages(snapshot: dict[str, object]) -> list[dict[str, str]]:
    """Render the exact chat messages supplied to the current Qwen prompt version."""
    return [
        {
            "role": "system",
            "content": (
                "You write concise, evidence-grounded academic Related Work prose. "
                "Return 1-2 connected paragraphs of about 150-220 words. Compare approaches "
                "instead of restating abstracts. Use only relevant [R1]-[R5] citations, and "
                "never output headings, lists, evidence labels, or metadata."
            ),
        },
        {"role": "user", "content": build_qwen_prompt(snapshot)},
    ]


def _has_repeated_long_sequence(text: str, width: int = 30) -> bool:
    words = re.findall(r"\b\w+\b", text.casefold())
    seen: dict[tuple[str, ...], int] = {}
    for index in range(max(0, len(words) - width + 1)):
        sequence = tuple(words[index : index + width])
        previous = seen.get(sequence)
        if previous is not None and index - previous >= width:
            return True
        seen.setdefault(sequence, index)
    return False


def validate_generated_output(
    text: str, allowed: set[str], finish_reason: str, *, require_citation: bool
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    found = list(dict.fromkeys(re.findall(r"\[([A-Za-z0-9_.:-]+)\]", text)))
    valid = [citation for citation in found if citation in allowed]
    unknown = sorted(set(found) - allowed)
    if require_citation and not valid:
        errors.append("no valid citation key was generated")
    if unknown:
        errors.append(f"unknown citation keys: {', '.join(unknown)}")
    if INTERNAL_MARKER_PATTERN.search(text):
        errors.append("output contains internal evidence markers")
    if UUID_PATTERN.search(text):
        errors.append("output contains an internal UUID")
    if NON_PROSE_FORMAT_PATTERN.search(text):
        errors.append("output contains a heading, bullet, or numbered list")
    paragraphs = [
        re.sub(r"\s+", " ", item).strip().casefold() for item in re.split(r"\n\s*\n", text)
    ]
    substantial = [item for item in paragraphs if len(item) >= 80]
    if len(substantial) != len(set(substantial)) or _has_repeated_long_sequence(text):
        errors.append("output contains repeated paragraphs or long text sequences")
    if finish_reason == "length":
        errors.append("generation reached max_new_tokens before EOS")
    if text and text.rstrip()[-1] not in ".!?)]":
        errors.append("output ends with an incomplete sentence")
    return valid, errors


def validate_qwen_output(
    text: str, allowed: set[str], finish_reason: str
) -> tuple[list[str], list[str]]:
    citations, errors = validate_generated_output(
        text, allowed, finish_reason, require_citation=True
    )
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    if not 1 <= len(paragraphs) <= 2:
        errors.append(f"expected 1-2 paragraphs; generated {len(paragraphs)}")
    word_count = len(re.findall(r"\b[\w'-]+\b", text))
    if not 150 <= word_count <= 220:
        errors.append(f"expected 150-220 words; generated {word_count}")
    return citations, errors


def map_qwen_citations(
    text: str, snapshot: dict[str, object]
) -> tuple[str, list[str]]:
    """Replace validated short prompt IDs with persisted source citation keys."""
    mapping = qwen_citation_map(snapshot)
    found = list(dict.fromkeys(re.findall(r"\[([A-Za-z0-9_.:-]+)\]", text)))
    unknown = sorted(set(found) - set(mapping))
    if unknown:
        raise ValueError(f"Cannot map unknown Qwen citation IDs: {', '.join(unknown)}")
    mapped = re.sub(
        r"\[(R[1-5])\]",
        lambda match: f"[{mapping[match.group(1)]}]",
        text,
    )
    citations = list(dict.fromkeys(mapping[item] for item in found))
    if extract_citations(mapped, set(mapping.values())) != citations:
        raise RuntimeError("Mapped Qwen citations failed validation")
    return mapped, citations


class RelatedWorkGenerator(ABC):
    model_key: str
    repository_id: str
    context_limit: int

    def __init__(
        self,
        cache_dir: Path,
        *,
        device: Device,
        revision: str | None,
        max_new_tokens: int,
        safety_margin: int,
    ):
        self.cache_dir = cache_dir
        self.device = choose_device(device)
        self.revision = resolve_revision(self.repository_id, revision)
        self.max_new_tokens = max_new_tokens
        self.safety_margin = safety_margin

    @abstractmethod
    def generate(self, snapshot: dict[str, object]) -> GenerationOutput: ...

    def _validate_model_size(self, model) -> int:
        count = sum(parameter.numel() for parameter in model.parameters())
        if count >= MAX_MODEL_PARAMETERS:
            raise ValueError(f"{self.repository_id} has {count} parameters; must be <500M")
        return count


class PrimeraGenerator(RelatedWorkGenerator):
    model_key = "primera"
    repository_id = PRIMERA_REPOSITORY
    context_limit = 4096

    def __init__(
        self,
        *args,
        min_new_tokens: int = 56,
        num_beams: int = 4,
        length_penalty: float = 2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if min_new_tokens > self.max_new_tokens:
            raise ValueError("PRIMERA min_new_tokens cannot exceed max_new_tokens")
        self.min_new_tokens = min_new_tokens
        self.num_beams = num_beams
        self.length_penalty = length_penalty

    def generate(self, snapshot: dict[str, object]) -> GenerationOutput:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.repository_id, revision=self.revision, cache_dir=self.cache_dir
        )
        source = build_primera_input(snapshot)
        encoded = tokenizer(source, truncation=False, return_tensors="pt")
        count = int(encoded["input_ids"].shape[-1])
        digest = sha256(source.encode()).hexdigest()
        error = (
            f"context_overflow: encoder_input={count}, encoder_limit={self.context_limit}"
            if count > self.context_limit
            else None
        )
        if error:
            return GenerationOutput(
                "context_overflow", None, count, None, self.context_limit, digest, [], error
            )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.repository_id, revision=self.revision, cache_dir=self.cache_dir
        )
        self.parameter_count = self._validate_model_size(model)
        model.to(self.device).eval()
        docsep_token_id = tokenizer.convert_tokens_to_ids("<doc-sep>")
        if docsep_token_id in (None, tokenizer.unk_token_id):
            raise RuntimeError("PRIMERA tokenizer does not define <doc-sep>")
        global_attention_mask = torch.zeros_like(encoded["input_ids"])
        global_attention_mask[:, 0] = 1
        global_attention_mask[encoded["input_ids"] == docsep_token_id] = 1
        with torch.inference_mode():
            generated = model.generate(
                **{k: v.to(self.device) for k, v in encoded.items()},
                global_attention_mask=global_attention_mask.to(self.device),
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=self.min_new_tokens,
                num_beams=self.num_beams,
                no_repeat_ngram_size=3,
                length_penalty=self.length_penalty,
                use_cache=True,
                return_dict_in_generate=True,
            )
        sequence = generated.sequences[0]
        text = tokenizer.decode(sequence, skip_special_tokens=True).strip()
        if not text:
            raise RuntimeError("PRIMERA generated no text")
        eos_ids = model.generation_config.eos_token_id or tokenizer.eos_token_id
        eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
        ended_with_eos = bool(len(sequence) and int(sequence[-1]) in eos_ids)
        output_tokens = int(sequence.shape[-1])
        finish_reason = (
            "eos"
            if ended_with_eos
            else "length"
            if output_tokens >= self.max_new_tokens
            else "unknown"
        )
        allowed = {str(p["citation_key"]) for p in snapshot["recommended_papers"]}
        citations, validation_errors = validate_generated_output(
            text, allowed, finish_reason, require_citation=False
        )
        status = "invalid" if validation_errors else "completed"
        return GenerationOutput(
            status,
            text,
            count,
            output_tokens,
            self.context_limit,
            digest,
            citations,
            "; ".join(validation_errors) or None,
            finish_reason,
            tuple(validation_errors),
        )


class QwenGenerator(RelatedWorkGenerator):
    model_key = "qwen"
    repository_id = QWEN_REPOSITORY
    context_limit = 32768

    def _load_model(self):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            self.repository_id, revision=self.revision, cache_dir=self.cache_dir
        )

    def generate(self, snapshot: dict[str, object]) -> GenerationOutput:
        import torch
        from transformers import AutoTokenizer

        if len(snapshot["recommended_papers"]) != 5 or len(snapshot["chunk_pairs"]) != 15:
            raise ValueError("Qwen evidence must contain 5 papers and 15 chunk pairs")
        tokenizer = AutoTokenizer.from_pretrained(
            self.repository_id, revision=self.revision, cache_dir=self.cache_dir
        )
        messages = build_qwen_messages(snapshot)
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        count = int(encoded["input_ids"].shape[-1])
        digest = sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        error = validate_budget(count, self.max_new_tokens, self.safety_margin, self.context_limit)
        if error:
            return GenerationOutput(
                "context_overflow", None, count, None, self.context_limit, digest, [], error
            )
        model = self._load_model()
        self.parameter_count = self._validate_model_size(model)
        model.to(self.device).eval()
        generation_config = deepcopy(model.generation_config)
        generation_config.max_new_tokens = self.max_new_tokens
        generation_config.do_sample = False
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None
        generation_config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        generation_config.use_cache = True
        with torch.inference_mode():
            output = model.generate(
                **{k: v.to(self.device) for k, v in encoded.items()},
                generation_config=generation_config,
                return_dict_in_generate=True,
            )
        new_tokens = output.sequences[0, count:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        eos_ids = model.generation_config.eos_token_id or tokenizer.eos_token_id
        eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
        ended_with_eos = bool(len(new_tokens) and int(new_tokens[-1]) in eos_ids)
        finish_reason = (
            "eos"
            if ended_with_eos
            else "length"
            if len(new_tokens) >= self.max_new_tokens
            else "unknown"
        )
        short_ids = set(qwen_citation_map(snapshot))
        _short_citations, validation_errors = validate_qwen_output(
            text, short_ids, finish_reason
        )
        mapped_text = text
        citations: list[str] = []
        try:
            mapped_text, citations = map_qwen_citations(text, snapshot)
        except ValueError as exc:
            mapping_error = str(exc)
            if mapping_error not in validation_errors:
                validation_errors.append(mapping_error)
        status = "invalid" if validation_errors else "completed"
        return GenerationOutput(
            status,
            mapped_text,
            count,
            int(new_tokens.shape[-1]),
            self.context_limit,
            digest,
            citations,
            "; ".join(validation_errors) or None,
            finish_reason,
            tuple(validation_errors),
        )


class QwenLoraGenerator(QwenGenerator):
    """The existing Qwen inference path with one immutable PEFT adapter attached."""

    model_key = "qwen_lora"

    def __init__(self, *args, adapter_path: Path, adapter_revision: str, **kwargs):
        self.adapter_path = adapter_path
        self.adapter_revision = adapter_revision
        self.adapter_repository_id = str(adapter_path)
        super().__init__(*args, **kwargs)

    def _load_model(self):
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM

        peft_config = PeftConfig.from_pretrained(self.adapter_path)
        if peft_config.base_model_name_or_path != self.repository_id:
            raise ValueError(
                "LoRA adapter base model does not match Qwen/Qwen2.5-0.5B-Instruct"
            )
        base_model = AutoModelForCausalLM.from_pretrained(
            self.repository_id, revision=self.revision, cache_dir=self.cache_dir
        )
        return PeftModel.from_pretrained(
            base_model, self.adapter_path, is_trainable=False
        )
