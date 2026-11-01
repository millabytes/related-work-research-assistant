import numpy as np
import pytest

from related_work_assistant.embeddings.encoders import ModelDescriptor, validate_vectors
from related_work_assistant.pipelines.embeddings import model_key, paper_input, text_hash


def descriptor(config=None):
    return ModelDescriptor(
        role="paper_embedding",
        repository_id="test/model",
        checkpoint_revision="a" * 40,
        adapter_repository_id="test/adapter",
        adapter_revision="b" * 40,
        embedding_dim=3,
        parameter_count=10,
        config=config or {"normalized": False},
    )


def test_paper_input_and_hash_are_exact_and_stable():
    content = paper_input("  A title ", " An abstract.  ", "[SEP]")
    assert content == "A title[SEP]An abstract."
    assert text_hash(content) == text_hash("A title[SEP]An abstract.")
    assert text_hash(content) != text_hash(content + " ")


def test_model_key_changes_with_reproducibility_inputs():
    first = model_key(descriptor())
    assert first == model_key(descriptor())
    assert first != model_key(descriptor({"normalized": True}))


def test_vector_validation():
    vectors = validate_vectors(np.ones((2, 3)), 2, 3)
    assert vectors.dtype == np.float32
    with pytest.raises(ValueError, match="shape"):
        validate_vectors(np.ones((2, 4)), 2, 3)
    with pytest.raises(ValueError, match="NaN"):
        validate_vectors(np.array([[np.nan, 1, 2]]), 1, 3)
    with pytest.raises(ValueError, match="zero"):
        validate_vectors(np.zeros((1, 3)), 1, 3)
