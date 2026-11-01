from related_work_assistant.retrieval.chunks import near_duplicate


def test_near_duplicate_uses_word_shingles():
    original = (
        "Neural models learn useful latent representations from scientific input data "
        "without direct supervision from labels."
    )
    almost_same = original + " "
    different = "The experiment evaluates classification accuracy on three external datasets."
    assert near_duplicate(original, almost_same)
    assert not near_duplicate(original, different)


def test_short_text_duplicate_is_normalized():
    assert near_duplicate("  SAME text ", "same TEXT")
    assert not near_duplicate("first", "second")
