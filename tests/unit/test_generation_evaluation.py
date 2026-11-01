import json

import pytest

from related_work_assistant.evaluation.generation.benchmark import (
    _model_summary,
    sample_test_examples,
)
from related_work_assistant.evaluation.generation.metrics import (
    citation_metrics,
    link_claims_to_evidence,
    normalize_benchmark_citations,
    rouge_scores,
)
from related_work_assistant.evaluation.generation.nli import score_grounding_links


def test_rouge_normalizes_citation_markers_but_preserves_original_text():
    prediction = "Prior work [R1] uses graphs."
    reference = "Prior work @cite_42 uses graphs."

    assert normalize_benchmark_citations(prediction) == "Prior work  <cite>  uses graphs."
    assert rouge_scores(prediction, reference) == {
        "rouge1": 1.0,
        "rouge2": 1.0,
        "rougeL": 1.0,
    }
    assert prediction == "Prior work [R1] uses graphs."


def test_rouge_f1_uses_multiset_overlap_and_lcs():
    scores = rouge_scores("a b c", "a b d")
    assert scores["rouge1"] == pytest.approx(2 / 3)
    assert scores["rouge2"] == pytest.approx(0.5)
    assert scores["rougeL"] == pytest.approx(2 / 3)


def test_citation_validity_coverage_and_primera_unresolved():
    metrics = citation_metrics("Claim [a] and [invented], then [a].", {"a", "b"})
    assert metrics["validity"] == pytest.approx(2 / 3)
    assert metrics["coverage"] == pytest.approx(0.5)
    assert metrics["valid_citations"] == ["a"]
    assert metrics["invalid_citations"] == ["invented"]
    unresolved = citation_metrics("generic cite", {"a"}, unresolved=True)
    assert unresolved["status"] == "unresolved"
    assert unresolved["validity"] is None and unresolved["coverage"] is None


def test_claims_link_only_to_the_explicitly_cited_paper():
    snapshot = {
        "recommended_papers": [
            {"paper_id": "paper-a", "citation_key": "a"},
            {"paper_id": "paper-b", "citation_key": "b"},
        ],
        "chunk_catalog": [
            {"chunk_id": "chunk-a", "text": "Evidence A"},
            {"chunk_id": "chunk-b", "text": "Evidence B"},
        ],
        "chunk_pairs": [
            {"recommended_paper_id": "paper-a", "recommended_chunk_id": "chunk-a"},
            {"recommended_paper_id": "paper-b", "recommended_chunk_id": "chunk-b"},
        ],
    }

    links = link_claims_to_evidence("A grounded claim [a].", snapshot)

    assert len(links) == 1
    assert links[0]["source_paper_id"] == "paper-a"
    assert [item["text"] for item in links[0]["evidence"]] == ["Evidence A"]
    assert "Evidence B" not in json.dumps(links)


def test_nli_selects_best_entailment_within_the_linked_source_only():
    links = [
        {
            "claim_index": 0,
            "text": "A claim [a].",
            "citations": ["a"],
            "citation": "a",
            "source_paper_id": "paper-a",
            "status": "linked",
            "evidence": [
                {"chunk_id": "low", "text": "Low evidence"},
                {"chunk_id": "high", "text": "High evidence"},
            ],
        }
    ]

    class Scorer:
        def score(self, pairs):
            assert [premise for premise, _claim in pairs] == ["Low evidence", "High evidence"]
            return [
                {"entailment": 0.2, "neutral": 0.7, "contradiction": 0.1, "label": "neutral"},
                {"entailment": 0.8, "neutral": 0.1, "contradiction": 0.1, "label": "entailment"},
            ]

    scored, aggregate = score_grounding_links(links, Scorer())

    assert scored[0]["best_evidence"]["chunk_id"] == "high"
    assert aggregate["entailment"] == pytest.approx(0.8)
    assert aggregate["evaluated_claim_sources"] == 1


def test_benchmark_sampling_is_reproducible_and_keeps_concrete_ids(tmp_path):
    path = tmp_path / "test.jsonl"
    rows = [
        {"example_id": f"paper-{index}", "split_index": index}
        for index in range(20)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    first = sample_test_examples(path, limit=5, seed=17)
    second = sample_test_examples(path, limit=5, seed=17)
    different = sample_test_examples(path, limit=5, seed=18)

    assert [row["benchmark_sample_id"] for row in first] == [
        row["benchmark_sample_id"] for row in second
    ]
    assert [row["benchmark_sample_id"] for row in first] != [
        row["benchmark_sample_id"] for row in different
    ]


def test_unavailable_benchmark_model_is_not_reported_as_zero():
    examples = [
        {"predictions": {"qwen_lora": {"status": "unavailable", "rouge": None}}}
    ]

    summary = _model_summary(examples, "qwen_lora")

    assert summary["status"] == "unavailable"
    assert summary["evaluated"] == 0
    assert summary["rouge1"] is None
