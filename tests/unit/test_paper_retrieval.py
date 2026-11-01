import pytest

from related_work_assistant.retrieval.papers import (
    InvalidPaperFilterError,
    _literal_contains_pattern,
    validate_filters,
)


def test_filter_validation_and_normalization():
    assert validate_filters(2010, 2020, "  NeurIPS  ") == "NeurIPS"
    assert validate_filters(None, None, None) is None
    with pytest.raises(InvalidPaperFilterError, match="year_from"):
        validate_filters(2021, 2020, None)
    with pytest.raises(InvalidPaperFilterError, match="blank"):
        validate_filters(None, None, "  ")


def test_venue_pattern_treats_wildcards_as_literals():
    assert _literal_contains_pattern("100%_AI") == "%100\\%\\_AI%"
