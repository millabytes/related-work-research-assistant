import json
from hashlib import sha256

import peft
import pytest
import transformers

from related_work_assistant.core.config import Settings
from related_work_assistant.generation.generators import QWEN_REPOSITORY, QwenLoraGenerator
from related_work_assistant.generation.service import (
    AVAILABLE_MODELS,
    DEFAULT_MODELS,
    model_config,
    request_cache_key,
)
from related_work_assistant.training.lora_artifacts import (
    adapter_fingerprint,
    adapter_status,
)
from related_work_assistant.training.qwen_lora import (
    AssistantOnlyCollator,
    JsonlChatDataset,
)


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        rendered = "".join(f"<{row['role']}>{row['content']}" for row in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        return list(rendered.encode())


def _write_adapter(root, weights=b"adapter-v1"):
    adapter = root / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": QWEN_REPOSITORY})
    )
    (adapter / "adapter_model.safetensors").write_bytes(weights)
    revision = adapter_fingerprint(adapter)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "adapter_revision": revision,
                "base_model_id": QWEN_REPOSITORY,
                "base_model_revision": "a" * 40,
            }
        )
    )
    return adapter, revision


def test_training_labels_mask_every_prompt_token_and_collator_masks_padding(tmp_path):
    prompt = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "input [R1]"},
    ]
    row = {
        "split_index": 0,
        "input_messages": prompt,
        "messages": [*prompt, {"role": "assistant", "content": "target [R1]."}],
    }
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(row) + "\n")
    dataset = JsonlChatDataset(path, FakeTokenizer(), limit=None, max_tokens=32768)

    encoded = dataset[0]
    boundary = len(
        FakeTokenizer().apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True
        )
    )
    assert encoded["labels"][:boundary] == [-100] * boundary
    assert encoded["labels"][boundary:] == encoded["input_ids"][boundary:]
    assert all(label != -100 for label in encoded["labels"][boundary:])

    batch = AssistantOnlyCollator(0)([encoded, {key: value[:-3] for key, value in encoded.items()}])
    assert batch["labels"][1, -3:].tolist() == [-100, -100, -100]
    assert batch["attention_mask"][1, -3:].tolist() == [0, 0, 0]


def test_adapter_availability_and_cache_key_include_adapter_revision(tmp_path, monkeypatch):
    assert "qwen_lora" in AVAILABLE_MODELS
    assert "qwen_lora" not in DEFAULT_MODELS
    unavailable = adapter_status(tmp_path)
    assert unavailable["available"] is False
    adapter, first_revision = _write_adapter(tmp_path)
    assert adapter_status(tmp_path)["available"] is True
    settings = Settings(
        _env_file=None,
        postgres_password="test",
        qwen_lora_dir=tmp_path,
    )
    config = model_config(settings, "qwen_lora")
    assert config["adapter_revision"] == first_revision
    assert config["requested_revision"] == "a" * 40
    monkeypatch.setattr(
        "related_work_assistant.generation.service.evidence_state_hash", lambda *_args: "e" * 64
    )
    first_key = request_cache_key(None, settings, "paper-id", "qwen_lora")

    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-v2")
    second_revision = adapter_fingerprint(adapter)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "adapter_revision": second_revision,
                "base_model_id": QWEN_REPOSITORY,
                "base_model_revision": "a" * 40,
            }
        )
    )
    second_key = request_cache_key(None, settings, "paper-id", "qwen_lora")

    assert first_revision != second_revision
    assert first_key != second_key
    assert len(first_key) == len(sha256().hexdigest())


def test_qwen_lora_loads_peft_adapter_over_the_base_model(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    base_model = object()
    wrapped_model = object()

    class Config:
        base_model_name_or_path = QWEN_REPOSITORY

    monkeypatch.setattr(peft.PeftConfig, "from_pretrained", lambda path: Config())
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda repository, **kwargs: base_model,
    )
    calls = []

    def fake_peft_load(model, path, *, is_trainable):
        calls.append((model, path, is_trainable))
        return wrapped_model

    monkeypatch.setattr(peft.PeftModel, "from_pretrained", fake_peft_load)
    generator = object.__new__(QwenLoraGenerator)
    generator.adapter_path = adapter
    generator.repository_id = QWEN_REPOSITORY
    generator.revision = "a" * 40
    generator.cache_dir = tmp_path

    assert generator._load_model() is wrapped_model
    assert calls == [(base_model, adapter, False)]


def test_qwen_lora_rejects_adapter_for_another_base_model(tmp_path, monkeypatch):
    class Config:
        base_model_name_or_path = "another/model"

    monkeypatch.setattr(peft.PeftConfig, "from_pretrained", lambda path: Config())
    generator = object.__new__(QwenLoraGenerator)
    generator.adapter_path = tmp_path
    generator.repository_id = QWEN_REPOSITORY

    with pytest.raises(ValueError, match="base model does not match"):
        generator._load_model()
