"""Tests for src.event_formation.event_pipeline.load_clean_detections."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.event_formation.event_pipeline import REQUIRED_DETECTION_COLUMNS, load_clean_detections


def _write_minimal_clean_csv(path: Path) -> Path:
    df = pd.DataFrame(
        {
            "latitude": [21.5, 21.6],
            "longitude": [82.1, 82.2],
            "acq_datetime": ["2023-01-01T06:55:00+00:00", "2023-01-01T07:10:00+00:00"],
            "frp": [2.89, 3.5],
            "frp_valid": [True, True],
            "bright_ti4": [330.0, 331.0],
            "bright_ti5": [290.0, 291.0],
            "confidence": ["n", "l"],
            "daynight": ["D", "D"],
        }
    )
    df.to_csv(path, index=False)
    return path


def test_load_clean_detections_parses_utc_datetime(tmp_path: Path) -> None:
    path = _write_minimal_clean_csv(tmp_path / "clean.csv")
    df = load_clean_detections(path)

    assert str(df["acq_datetime"].dt.tz) == "UTC"
    assert len(df) == 2


def test_load_clean_detections_respects_max_rows(tmp_path: Path) -> None:
    path = _write_minimal_clean_csv(tmp_path / "clean.csv")
    df = load_clean_detections(path, max_rows=1)
    assert len(df) == 1


def test_load_clean_detections_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_clean_detections(tmp_path / "missing.csv")


def test_load_clean_detections_missing_columns_raises(tmp_path: Path) -> None:
    df = pd.DataFrame({"latitude": [1.0], "longitude": [2.0]})
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)

    with pytest.raises(ValueError) as exc_info:
        load_clean_detections(path)

    assert "acq_datetime" in str(exc_info.value)


def test_required_columns_constant_matches_expected_fields() -> None:
    assert set(REQUIRED_DETECTION_COLUMNS) == {
        "latitude",
        "longitude",
        "acq_datetime",
        "frp",
        "frp_valid",
        "bright_ti4",
        "bright_ti5",
        "confidence",
        "daynight",
    }
