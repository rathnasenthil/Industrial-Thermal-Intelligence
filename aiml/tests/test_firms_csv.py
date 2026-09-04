"""Tests for src.data_ingestion.firms_csv (loading, schema validation, provenance)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_ingestion.firms_csv import (
    REQUIRED_COLUMNS,
    FirmsSchemaError,
    load_firms_csv,
    load_firms_csv_files,
)

_VALID_ROW = {
    "latitude": "17.99205",
    "longitude": "82.99986",
    "bright_ti4": "332.1",
    "scan": "0.5",
    "track": "0.66",
    "acq_date": "2023-01-01",
    "acq_time": "655",
    "satellite": "N20",
    "instrument": "VIIRS",
    "confidence": "n",
    "version": "2",
    "bright_ti5": "292.17",
    "frp": "2.89",
    "daynight": "D",
    "type": "0",
}


def _write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_firms_csv_adds_source_file(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "firms_viirs_india_2023.csv", [_VALID_ROW])
    df = load_firms_csv(csv_path)
    assert "source_file" in df.columns
    assert (df["source_file"] == "firms_viirs_india_2023.csv").all()


def test_load_firms_csv_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_firms_csv(tmp_path / "does_not_exist.csv")


def test_load_firms_csv_missing_columns_raises_clear_error(tmp_path: Path) -> None:
    row = dict(_VALID_ROW)
    del row["frp"]
    del row["confidence"]
    csv_path = _write_csv(tmp_path / "bad.csv", [row])

    with pytest.raises(FirmsSchemaError) as exc_info:
        load_firms_csv(csv_path)

    message = str(exc_info.value)
    assert "frp" in message
    assert "confidence" in message


def test_load_firms_csv_does_not_mutate_original_file(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "firms_viirs_india_2023.csv", [_VALID_ROW])
    original_bytes = csv_path.read_bytes()

    load_firms_csv(csv_path)

    assert csv_path.read_bytes() == original_bytes


def test_load_firms_csv_files_combines_and_preserves_source_file(tmp_path: Path) -> None:
    row_2023 = dict(_VALID_ROW)
    row_2024 = dict(_VALID_ROW, acq_date="2024-01-01")

    path_2023 = _write_csv(tmp_path / "firms_viirs_india_2023.csv", [row_2023])
    path_2024 = _write_csv(tmp_path / "firms_viirs_india_2024.csv", [row_2024])

    combined = load_firms_csv_files([path_2023, path_2024])

    assert len(combined) == 2
    assert set(combined["source_file"]) == {
        "firms_viirs_india_2023.csv",
        "firms_viirs_india_2024.csv",
    }
    for col in REQUIRED_COLUMNS:
        assert col in combined.columns


def test_load_firms_csv_files_empty_list_raises() -> None:
    with pytest.raises(ValueError):
        load_firms_csv_files([])
