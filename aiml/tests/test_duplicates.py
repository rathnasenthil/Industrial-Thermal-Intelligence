"""Tests for src.preprocessing.duplicates (exact-duplicate detection)."""

from __future__ import annotations

import pandas as pd

from src.preprocessing.duplicates import detect_exact_duplicates

_SUBSET = ["latitude", "longitude", "acq_date", "acq_time", "frp"]


def test_no_duplicates_when_all_rows_distinct() -> None:
    df = pd.DataFrame(
        {
            "latitude": [17.99, 18.97, 20.83],
            "longitude": [83.0, 83.8, 86.9],
            "acq_date": ["2023-01-01"] * 3,
            "acq_time": [655, 655, 656],
            "frp": [2.89, 2.41, 7.56],
        }
    )

    result = detect_exact_duplicates(df, subset=_SUBSET)

    assert result.stats["exact_duplicate_count"] == 0
    assert not result.duplicate_mask.any()


def test_exact_duplicate_row_is_detected() -> None:
    df = pd.DataFrame(
        {
            "latitude": [17.99, 17.99, 20.83],
            "longitude": [83.0, 83.0, 86.9],
            "acq_date": ["2023-01-01", "2023-01-01", "2023-01-01"],
            "acq_time": [655, 655, 656],
            "frp": [2.89, 2.89, 7.56],
        }
    )

    result = detect_exact_duplicates(df, subset=_SUBSET)

    assert result.stats["exact_duplicate_count"] == 1
    assert list(result.duplicate_mask) == [False, True, False]


def test_same_location_different_time_is_not_a_duplicate() -> None:
    """A persistent thermal source re-detected on different overpasses is NOT a duplicate."""
    df = pd.DataFrame(
        {
            "latitude": [17.99, 17.99],
            "longitude": [83.0, 83.0],
            "acq_date": ["2023-01-01", "2023-01-02"],
            "acq_time": [655, 701],
            "frp": [2.89, 3.10],
        }
    )

    result = detect_exact_duplicates(df, subset=_SUBSET)

    assert result.stats["exact_duplicate_count"] == 0


def test_same_location_and_time_different_frp_is_not_a_duplicate() -> None:
    """Adjacent VIIRS pixels from the same large fire share a timestamp but differ in FRP."""
    df = pd.DataFrame(
        {
            "latitude": [17.99, 17.99],
            "longitude": [83.0, 83.0],
            "acq_date": ["2023-01-01", "2023-01-01"],
            "acq_time": [655, 655],
            "frp": [2.89, 15.4],
        }
    )

    result = detect_exact_duplicates(df, subset=_SUBSET)

    assert result.stats["exact_duplicate_count"] == 0


def test_strategy_note_explains_why_lat_lon_alone_is_not_used() -> None:
    df = pd.DataFrame({"latitude": [1.0], "longitude": [2.0], "acq_date": ["2023-01-01"], "acq_time": [0], "frp": [1.0]})
    result = detect_exact_duplicates(df, subset=_SUBSET)
    assert "latitude/longitude alone" in result.strategy_note
