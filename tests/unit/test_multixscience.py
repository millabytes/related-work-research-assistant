import gzip
import json

from related_work_assistant.datasets.multixscience import (
    InvalidExample,
    PreparedExample,
    analyze_invalid_dataset,
    iter_json_array,
    prepare_dataset,
    prepare_example,
    summarize_invalid_examples,
)


class FakeQwenTokenizer:
    name_or_path = "fake-qwen-tokenizer"

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
        assert tokenize is True
        rendered = ""
        for message in conversation:
            rendered += f"<{message['role']}>\n{message['content']}\n"
        if add_generation_prompt:
            rendered += "<assistant>\n"
        return list(rendered.encode())


def _raw_example():
    return {
        "aid": "paper-a",
        "mid": "123",
        "abstract": "A target abstract.",
        "related_work": "Prior sentinel work @cite_11 differs from @cite_5.",
        "ref_abstract": {
            "@cite_11": {"mid": "ref-11", "abstract": "Second reference."},
            "@cite_5": {"mid": "ref-5", "abstract": "First reference."},
        },
    }


def test_citations_are_mapped_by_reference_order_and_originals_preserved():
    result = prepare_example(
        _raw_example(), split="train", split_index=7, tokenizer=FakeQwenTokenizer()
    )

    assert isinstance(result, PreparedExample)
    assert result.citation_mapping == {"@cite_5": "[R1]", "@cite_11": "[R2]"}
    assert result.assistant_target == "Prior sentinel work [R2] differs from [R1]."
    assert result.original_related_work == _raw_example()["related_work"]
    assert result.original_citation_markers == ["@cite_11", "@cite_5"]
    assert [row["mid"] for row in result.references] == ["ref-5", "ref-11"]


def test_target_is_not_present_in_input_and_assistant_boundary_is_recorded():
    result = prepare_example(
        _raw_example(), split="validation", split_index=0, tokenizer=FakeQwenTokenizer()
    )

    assert isinstance(result, PreparedExample)
    input_text = json.dumps(result.input_messages)
    assert "Prior sentinel work" not in input_text
    assert result.messages[-1] == {"role": "assistant", "content": result.assistant_target}
    assert result.loss_scope == "assistant_only"
    assert result.assistant_token_start == result.input_token_count
    assert result.assistant_token_count == result.total_token_count - result.input_token_count


def test_missing_citation_reference_and_token_overflow_are_invalid():
    missing = _raw_example()
    del missing["ref_abstract"]["@cite_11"]
    missing_result = prepare_example(
        missing, split="test", split_index=0, tokenizer=FakeQwenTokenizer()
    )
    overflow_result = prepare_example(
        _raw_example(),
        split="test",
        split_index=1,
        tokenizer=FakeQwenTokenizer(),
        max_sequence_tokens=10,
    )

    assert isinstance(missing_result, InvalidExample)
    assert "target citations missing from references: @cite_11" in missing_result.reasons
    assert isinstance(overflow_result, InvalidExample)
    assert overflow_result.reasons[0].startswith("token overflow:")


def test_invalid_statistics_count_examples_not_individual_markers():
    rows = [
        InvalidExample(
            "a",
            "train",
            0,
            [
                "cited reference @cite_1 has no abstract",
                "cited reference @cite_2 has no abstract",
            ],
            ["@cite_1", "@cite_2"],
        ),
        InvalidExample(
            "b",
            "validation",
            1,
            ["target citations missing from references: @cite_9"],
            ["@cite_9"],
        ),
    ]

    summary = summarize_invalid_examples(rows)

    assert summary["invalid_by_reason"] == {
        "cited_reference_missing_abstract": 1,
        "target_citation_missing_from_references": 1,
    }
    sample = summary["invalid_examples_by_reason"]["cited_reference_missing_abstract"][0]
    assert sample["example_id"] == "a"
    assert sample["original_citation_markers"] == ["@cite_1", "@cite_2"]


def test_streaming_json_array_parser_handles_small_chunks(tmp_path):
    path = tmp_path / "split.json.gz"
    expected = [_raw_example(), {**_raw_example(), "aid": "paper-b"}]
    with gzip.open(path, "wt", encoding="utf-8") as file:
        json.dump(expected, file)

    assert list(iter_json_array(path, chunk_size=37)) == expected


def test_prepare_dataset_preserves_split_identity_and_writes_statistics(tmp_path):
    raw_paths = {}
    for split in ("train", "validation", "test"):
        path = tmp_path / f"{split}.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as file:
            json.dump([{**_raw_example(), "aid": f"{split}-paper"}], file)
        raw_paths[split] = path

    result = prepare_dataset(
        dataset_dir=tmp_path / "prepared",
        tokenizer=FakeQwenTokenizer(),
        limit=1,
        raw_paths=raw_paths,
    )

    assert result["statistics"]["total_prepared"] == 3
    for split in raw_paths:
        row = json.loads((tmp_path / "prepared" / f"{split}.jsonl").read_text())
        assert row["split"] == split
        assert row["example_id"] == f"{split}-paper"
    config = json.loads((tmp_path / "prepared/preprocessing_config.json").read_text())
    assert config["loss_scope"] == "assistant_only"
    assert config["silent_truncation"] is False


def test_invalid_audit_finds_safe_exact_mid_recovery(tmp_path):
    invalid = {
        "aid": "invalid-paper",
        "mid": "target-mid",
        "abstract": "Target abstract.",
        "related_work": "Missing evidence @cite_1.",
        "ref_abstract": {"@cite_1": {"mid": "recoverable-mid", "abstract": ""}},
    }
    recovery_source = {
        **_raw_example(),
        "aid": "source-paper",
        "mid": "recoverable-mid",
        "abstract": "The exact referenced paper abstract.",
    }
    raw_paths = {}
    values = {"train": invalid, "validation": recovery_source, "test": _raw_example()}
    dataset_dir = tmp_path / "prepared"
    raw_dir = dataset_dir / "raw"
    raw_dir.mkdir(parents=True)
    filenames = {"train": "train.json.gz", "validation": "val.json.gz", "test": "test.json.gz"}
    for split, value in values.items():
        path = raw_dir / filenames[split]
        with gzip.open(path, "wt", encoding="utf-8") as file:
            json.dump([value], file)
        raw_paths[split] = path
    prepare_dataset(
        dataset_dir=dataset_dir,
        tokenizer=FakeQwenTokenizer(),
        raw_paths=raw_paths,
    )

    statistics = analyze_invalid_dataset(dataset_dir)

    assert statistics["invalid_by_reason"] == {"cited_reference_missing_abstract": 1}
    audit = statistics["citation_recovery_audit"]["total"]
    assert audit["invalid_examples_only_missing_reference_abstract"] == 1
    assert audit["missing_cited_reference_occurrences"] == 1
    assert audit["examples_all_missing_abstracts_recoverable_by_exact_mid"] == 1
