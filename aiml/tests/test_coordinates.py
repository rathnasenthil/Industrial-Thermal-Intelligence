"""Tests for src.preprocessing.coordinates (latitude/longitude validation)."""

from __future__ import annotations

import pandas as pd

from src.preprocessing.coordinates import validate_coordinates


def test_valid_coordinates_all_pass() -> None:
    df = pd.DataFrame({"latitude": [17.99, -45.0, 0.0, 90.0, -90.0], "longitude": [83.0, 179.9, -180.0, 0.0, 180.0]})

    result = validate_coordinates(df)

    assert result.valid_mask.all()
    assert result.stats["invalid_count"] == 0
    assert result.stats["missing_count"] == 0
    assert result.stats["out_of_range_count"] == 0


def test_out_of_range_coordinates_are_flagged_invalid() -> None:
    df = pd.DataFrame(
        {
            "latitude": [17.99, 95.0, -91.0, 45.0],
            "longitude": [83.0, 200.0, 10.0, -200.0],
        }
    )

    result = validate_coordinates(df)

    assert list(result.valid_mask) == [True, False, False, False]
    assert result.stats["out_of_range_count"] == 3
    assert result.stats["invalid_count"] == 3
    assert result.stats["missing_count"] == 0


def test_missing_coordinates_are_flagged_invalid_and_counted_separately() -> None:
    df = pd.DataFrame({"latitude": [17.99, float("nan")], "longitude": [83.0, 10.0]})

    result = validate_coordinates(df)

    assert list(result.valid_mask) == [True, False]
    assert result.stats["missing_count"] == 1
    assert result.stats["out_of_range_count"] == 0
    assert result.stats["invalid_count"] == 1


def test_boundary_values_are_valid() -> None:
    """Exact boundary values (+/-90 lat, +/-180 lon) must be valid, not invalid."""
    df = pd.DataFrame({"latitude": [90.0, -90.0], "longitude": [180.0, -180.0]})

    result = validate_coordinates(df)

    assert result.valid_mask.all()
