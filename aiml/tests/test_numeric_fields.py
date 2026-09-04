"""Tests for src.preprocessing.numeric_fields (numeric conversion, FRP handling)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing.numeric_fields import convert_numeric_columns, validate_frp


def test_convert_numeric_columns_parses_valid_values() -> None:
    df = pd.DataFrame({"frp": ["2.89", "0.5", "100"], "scan": ["0.5", "0.43", "0.49"]})

    result = convert_numeric_columns(df, columns=("frp", "scan"))

    assert result.df["frp"].tolist() == [2.89, 0.5, 100.0]
    assert result.stats["frp"] == {"missing": 0, "invalid_non_numeric": 0}


def test_convert_numeric_columns_reports_missing_without_inventing_values() -> None:
    df = pd.DataFrame({"frp": ["2.89", "", None]})

    result = convert_numeric_columns(df, columns=("frp",))

    # Missing values must remain NaN, never replaced with 0 or any other value.
    assert result.df["frp"].iloc[0] == 2.89
    assert pd.isna(result.df["frp"].iloc[1])
    assert pd.isna(result.df["frp"].iloc[2])
    assert result.stats["frp"]["missing"] == 2
    assert result.stats["frp"]["invalid_non_numeric"] == 0


def test_convert_numeric_columns_reports_non_numeric_garbage_separately_from_missing() -> None:
    df = pd.DataFrame({"frp": ["2.89", "not-a-number", ""]})

    result = convert_numeric_columns(df, columns=("frp",))

    assert result.stats["frp"]["missing"] == 1
    assert result.stats["frp"]["invalid_non_numeric"] == 1
    assert pd.isna(result.df["frp"].iloc[1])


def test_validate_frp_missing_is_not_treated_as_zero() -> None:
    df = pd.DataFrame({"frp": [np.nan, 2.89]})

    result = validate_frp(df)

    assert list(result.valid_mask) == [False, True]
    assert result.stats["missing_or_unparsable"] == 1
    # Never silently replaced with 0.0 anywhere upstream.
    assert pd.isna(df["frp"].iloc[0])


def test_validate_frp_keeps_legitimate_low_values_as_valid() -> None:
    df = pd.DataFrame({"frp": [0.1, 0.0, 0.01]})

    result = validate_frp(df)

    # Small (even zero) FRP is a legitimate observation, not invalid data.
    assert result.valid_mask.all()
    assert result.stats["invalid_count"] == 0
    assert result.stats["valid_low_count"] == 3


def test_validate_frp_flags_negative_as_invalid() -> None:
    df = pd.DataFrame({"frp": [-1.0, 2.0]})

    result = validate_frp(df)

    assert list(result.valid_mask) == [False, True]
    assert result.stats["negative"] == 1
