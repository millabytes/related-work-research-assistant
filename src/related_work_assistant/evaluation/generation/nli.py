"""Small local NLI cross-encoder and source-constrained grounding aggregation."""

from typing import Any

from related_work_assistant.embeddings.encoders import Device, choose_device, resolve_revision

NLI_REPOSITORY = "cross-encoder/nli-MiniLM2-L6-H768"
NLI_MAX_LENGTH = 512


class LocalNliScorer:
    def __init__(
        self,
        *,
        cache_dir,
        device: Device = "auto",
        revision: str | None = None,
        batch_size: int = 16,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = choose_device(device)
        self.revision = resolve_revision(NLI_REPOSITORY, revision)
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            NLI_REPOSITORY, revision=self.revision, cache_dir=cache_dir
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            NLI_REPOSITORY, revision=self.revision, cache_dir=cache_dir
        ).to(self.device)
        self.model.eval()
        self.torch = torch
        labels = {
            int(index): str(label).casefold()
            for index, label in self.model.config.id2label.items()
        }
        if set(labels.values()) != {"entailment", "neutral", "contradiction"}:
            raise RuntimeError(f"Unexpected NLI labels: {labels}")
        self.labels = labels

    def score(self, pairs: list[tuple[str, str]]) -> list[dict[str, float | str]]:
        results: list[dict[str, float | str]] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            encoded = self.tokenizer(
                [premise for premise, _hypothesis in batch],
                [hypothesis for _premise, hypothesis in batch],
                padding=True,
                truncation=True,
                max_length=NLI_MAX_LENGTH,
                return_tensors="pt",
            )
            with self.torch.inference_mode():
                logits = self.model(**{key: value.to(self.device) for key, value in encoded.items()}).logits
                probabilities = self.torch.softmax(logits, dim=-1).cpu().tolist()
            for values in probabilities:
                scores = {self.labels[index]: float(value) for index, value in enumerate(values)}
                scores["label"] = max(
                    ("entailment", "neutral", "contradiction"), key=lambda key: scores[key]
                )
                results.append(scores)
        return results


def score_grounding_links(
    links: list[dict[str, Any]], scorer: Any
) -> tuple[list[dict[str, Any]], dict[str, float | int | None]]:
    """Choose best entailment only among passages of the explicitly cited source."""
    scored_links: list[dict[str, Any]] = []
    for link in links:
        if link["status"] != "linked":
            scored_links.append({**link, "nli": None, "evidence_scores": []})
            continue
        hypothesis = link.get("nli_hypothesis", link["text"])
        pairs = [(item["text"], hypothesis) for item in link["evidence"]]
        scores = scorer.score(pairs)
        evidence_scores = [
            {**evidence, "nli": score}
            for evidence, score in zip(link["evidence"], scores, strict=True)
        ]
        best = max(evidence_scores, key=lambda item: float(item["nli"]["entailment"]))
        scored_links.append(
            {
                **link,
                "nli": best["nli"],
                "best_evidence": {key: value for key, value in best.items() if key != "nli"},
                "evidence_scores": evidence_scores,
            }
        )
    evaluated = [link["nli"] for link in scored_links if link.get("nli") is not None]
    aggregate: dict[str, float | int | None] = {"evaluated_claim_sources": len(evaluated)}
    for label in ("entailment", "neutral", "contradiction"):
        aggregate[label] = (
            sum(float(item[label]) for item in evaluated) / len(evaluated) if evaluated else None
        )
    return scored_links, aggregate
