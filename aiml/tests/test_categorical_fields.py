"""Tests for src.preprocessing.categorical_fields (confidence/daynight preservation)."""

from __future__ import annotations

import pandas as pd

from src.preprocessing.categorical_fields import normalize_confidence, normalize_daynight


def test_confidence_letter_codes_are_preserved_exactly() -> None:
    """n/l/h must never be remapped to numbers or new categories."""
    series = pd.Series(["n", "l", "h"])
    result = normalize_confidence(series)
    assert result.tolist() == ["n", "l", "h"]


def test_confidence_whitespace_and_case_are_normalized_only() -> None:
    series = pd.Series([" n", "L", " H "])
    result = normalize_confidence(series)
    assert result.tolist() == ["n", "l", "h"]


def test_confidence_numeric_style_values_are_untouched_besides_whitespace() -> None:
    """MODIS-style numeric confidence (0-100) must not be reinterpreted."""
    series = pd.Series([" 45", "80", "12 "])
    result = normalize_confidence(series)
    assert result.tolist() == ["45", "80", "12"]


def test_daynight_is_preserved_with_only_formatting_normalization() -> None:
    series = pd.Series(["D", " n", "d"])
    result = normalize_daynight(series)
    assert result.tolist() == ["D", "N", "D"]
