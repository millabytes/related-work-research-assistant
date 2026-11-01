from uuid import uuid4

import pytest

from related_work_assistant.generation.evidence import canonical_hash
from related_work_assistant.generation.generators import (
    QWEN_REPOSITORY,
    QwenGenerator,
    build_primera_input,
    build_qwen_messages,
    build_qwen_prompt,
    extract_citations,
    map_qwen_citations,
    qwen_citation_map,
    validate_budget,
    validate_generated_output,
    validate_qwen_output,
)


def paper(index: int) -> dict[str, object]:
    return {
        "paper_id": str(uuid4()),
        "citation_key": f"source{index}",
        "title": f"Paper {index}",
        "authors": [f"Author {index}"],
        "year": 2020 + index,
        "venue": "Venue",
        "abstract": f"Complete abstract {index}.",
    }


def generation_snapshot() -> dict[str, object]:
    query = paper(0)
    recommendations = [paper(index) for index in range(1, 6)]
    chunks = []
    pairs = []
    for paper_index, recommendation in enumerate(recommendations, 1):
        for rank in range(1, 4):
            query_id = f"q-{paper_index}-{rank}"
            recommended_id = f"r-{paper_index}-{rank}"
            chunks.extend(
                [
                    {
                        "chunk_id": query_id,
                        "paper_id": query["paper_id"],
                        "text": f"Query text {paper_index} {rank}",
                        "section_heading": "Methods",
                        "section_type": "methods",
                    },
                    {
                        "chunk_id": recommended_id,
                        "paper_id": recommendation["paper_id"],
                        "text": f"Evidence text {paper_index} {rank}",
                        "section_heading": "Results",
                        "section_type": "results",
                    },
                ]
            )
            pairs.append(
                {
                    "recommended_paper_id": recommendation["paper_id"],
                    "rank": rank,
                    "query_chunk_id": query_id,
                    "recommended_chunk_id": recommended_id,
                    "bge_score": 0.8,
                    "reranking_score": 1.2,
                }
            )
    return {
        "query_paper": query,
        "recommended_papers": recommendations,
        "chunk_catalog": chunks,
        "chunk_pairs": pairs,
    }


def test_primera_uses_abstracts_and_expected_separator():
    snapshot = generation_snapshot()
    value = build_primera_input(snapshot)
    assert value.count("<doc-sep>") == 6
    assert value.startswith("Complete abstract 0.")
    assert value.endswith("<doc-sep>")
    assert "Complete abstract 0" in value and "Complete abstract 5" in value
    assert "Evidence text" not in value


def test_qwen_prompt_contains_all_five_papers_and_fifteen_pairs_verbatim():
    snapshot = generation_snapshot()
    prompt = build_qwen_prompt(snapshot)
    assert prompt.count("\nPaired evidence ") == 15
    query = snapshot["query_paper"]
    assert str(query["authors"][0]) not in prompt
    assert str(query["year"]) not in prompt
    assert str(query["venue"]) not in prompt
    for index, recommendation in enumerate(snapshot["recommended_papers"], 1):
        assert f"[R{index}]" in prompt
        assert str(recommendation["citation_key"]) not in prompt
        assert str(recommendation["paper_id"]) not in prompt
        assert str(recommendation["authors"][0]) not in prompt
        assert str(recommendation["year"]) not in prompt
        assert str(recommendation["venue"]) not in prompt
    for chunk in snapshot["chunk_catalog"]:
        assert chunk["text"] in prompt
        assert chunk["chunk_id"] not in prompt
    assert "CHUNK CATALOG" not in prompt
    assert "SECTION " not in prompt


def test_qwen_messages_include_system_rules_and_full_user_prompt():
    snapshot = generation_snapshot()
    messages = build_qwen_messages(snapshot)
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "1-2 connected paragraphs" in messages[0]["content"]
    assert messages[1]["content"] == build_qwen_prompt(snapshot)


def test_token_budget_and_citation_validation():
    assert validate_budget(30_000, 2048, 256, 32768) is None
    assert "context_overflow" in validate_budget(31_000, 2048, 256, 32768)
    assert extract_citations("Prior work [source1] and [source2].", {"source1", "source2"}) == [
        "source1",
        "source2",
    ]
    with pytest.raises(ValueError, match="unknown citation"):
        extract_citations("Fabricated [unknown].", {"source1"})


def test_evidence_hash_is_canonical_and_sensitive_to_input():
    first = {"b": 2, "a": [1, 3]}
    assert canonical_hash(first) == canonical_hash({"a": [1, 3], "b": 2})
    assert canonical_hash(first) != canonical_hash({"a": [1, 4], "b": 2})


def test_qwen_output_accepts_grounded_natural_completion():
    text = " ".join([*(f"comparison{index}" for index in range(154)), "[R1]."])
    citations, errors = validate_qwen_output(text, {"R1", "R2"}, "eos")
    assert citations == ["R1"]
    assert errors == []


def test_qwen_short_citations_map_to_original_keys():
    snapshot = generation_snapshot()
    assert qwen_citation_map(snapshot) == {
        "R1": "source1",
        "R2": "source2",
        "R3": "source3",
        "R4": "source4",
        "R5": "source5",
    }
    mapped, citations = map_qwen_citations(
        "A comparison [R2] extends another method [R1].", snapshot
    )
    assert mapped == "A comparison [source2] extends another method [source1]."
    assert citations == ["source2", "source1"]
    with pytest.raises(ValueError, match="unknown Qwen citation IDs"):
        map_qwen_citations("Unsupported [R6].", snapshot)


@pytest.mark.parametrize(
    ("text", "finish_reason", "expected"),
    [
        ("A comparison without a source.", "eos", "no valid citation"),
        ("A claim [invented].", "eos", "unknown citation keys"),
        ("SECTION Methods\nA claim [source1].", "eos", "internal evidence markers"),
        (
            "Internal 123e4567-e89b-12d3-a456-426614174000 [source1].",
            "eos",
            "internal UUID",
        ),
        ("A grounded but unfinished claim [source1]", "length", "max_new_tokens"),
        ("One short paragraph [source1].", "eos", "expected 150-220 words"),
        ("# Heading\n\nA claim [source1].", "eos", "heading, bullet, or numbered list"),
        ("An unfinished claim [source1", "eos", "incomplete sentence"),
    ],
)
def test_qwen_output_rejects_invalid_content(text, finish_reason, expected):
    _citations, errors = validate_qwen_output(text, {"source1"}, finish_reason)
    assert any(expected in error for error in errors)


def test_qwen_output_detects_repeated_long_sequences():
    repeated = " ".join(f"term{index}" for index in range(35))
    text = f"{repeated} [source1].\n\nA bridge sentence.\n\n{repeated} [source1]."
    _citations, errors = validate_qwen_output(text, {"source1"}, "eos")
    assert "output contains repeated paragraphs or long text sequences" in errors


def test_qwen_output_rejects_more_than_two_paragraphs():
    paragraphs = [
        " ".join(f"p{paragraph}word{index}" for index in range(55)) + " [R1]."
        for paragraph in range(3)
    ]
    _citations, errors = validate_qwen_output("\n\n".join(paragraphs), {"R1"}, "eos")
    assert "expected 1-2 paragraphs; generated 3" in errors


def test_primera_does_not_require_or_fabricate_citations():
    citations, errors = validate_generated_output(
        "A concise multi-document scientific summary.",
        {"source1"},
        "eos",
        require_citation=False,
    )
    assert citations == []
    assert errors == []


def test_qwen_repository_and_strict_parameter_limit():
    class Parameter:
        def __init__(self, count: int):
            self.count = count

        def numel(self) -> int:
            return self.count

    class Model:
        def __init__(self, count: int):
            self.count = count

        def parameters(self):
            return [Parameter(self.count)]

    generator = object.__new__(QwenGenerator)
    assert QWEN_REPOSITORY == "Qwen/Qwen2.5-0.5B-Instruct"
    assert generator._validate_model_size(Model(499_999_999)) == 499_999_999
    with pytest.raises(ValueError, match="must be <500M"):
        generator._validate_model_size(Model(500_000_000))
