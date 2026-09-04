"""Integration tests for the full FIRMS ingestion + preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing.firms_pipeline import run_firms_preprocessing, save_clean_dataset

_COLUMNS = [
    "latitude",
    "longitude",
    "bright_ti4",
    "scan",
    "track",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "confidence",
    "version",
    "bright_ti5",
    "frp",
    "daynight",
    "type",
]


def _row(**overrides) -> dict:
    base = {
        "latitude": 17.99205,
        "longitude": 82.99986,
        "bright_ti4": 332.1,
        "scan": 0.5,
        "track": 0.66,
        "acq_date": "2023-01-01",
        "acq_time": 655,
        "satellite": "N20",
        "instrument": "VIIRS",
        "confidence": "n",
        "version": 2,
        "bright_ti5": 292.17,
        "frp": 2.89,
        "daynight": "D",
        "type": 0,
    }
    base.update(overrides)
    return base


@pytest.fixture()
def two_year_dataset(tmp_path: Path) -> tuple[Path, Path]:
    rows_2023 = [
        _row(),
        _row(latitude=18.97135, longitude=83.80673, acq_time=656, frp=""),  # missing FRP
        _row(latitude=999.0, acq_time=700),  # invalid coordinate
        _row(acq_date="2023-01-01", acq_time=655),  # exact duplicate of first row
        _row(confidence="l", latitude=20.0, acq_time=712),
    ]
    rows_2024 = [
        _row(acq_date="2024-01-01", latitude=24.82792, longitude=93.75638, confidence="h"),
    ]

    path_2023 = tmp_path / "viirs-jpss1_2023_India.csv"
    path_2024 = tmp_path / "viirs-jpss1_2024_India.csv"
    pd.DataFrame(rows_2023, columns=_COLUMNS).to_csv(path_2023, index=False)
    pd.DataFrame(rows_2024, columns=_COLUMNS).to_csv(path_2024, index=False)
    return path_2023, path_2024


def test_pipeline_reports_expected_row_counts(two_year_dataset: tuple[Path, Path]) -> None:
    path_2023, path_2024 = two_year_dataset

    result = run_firms_preprocessing([path_2023, path_2024])

    counts = result.report["row_counts"]
    assert counts["total_input_rows"] == 6
    assert counts["combined_rows_before_cleaning"] == 6
    # 1 invalid coordinate row + 1 exact duplicate row = 2 rows removed.
    assert counts["invalid_rows_removed"] == 2
    assert counts["final_processed_rows"] == 4


def test_pipeline_preserves_source_file_provenance(two_year_dataset: tuple[Path, Path]) -> None:
    path_2023, path_2024 = two_year_dataset

    result = run_firms_preprocessing([path_2023, path_2024])

    assert set(result.clean_df["source_file"]) == {
        "viirs-jpss1_2023_India.csv",
        "viirs-jpss1_2024_India.csv",
    }
    year_2024_rows = result.clean_df[result.clean_df["source_file"] == "viirs-jpss1_2024_India.csv"]
    assert len(year_2024_rows) == 1


def test_pipeline_keeps_missing_frp_row_but_does_not_fill_it(two_year_dataset: tuple[Path, Path]) -> None:
    path_2023, path_2024 = two_year_dataset

    result = run_firms_preprocessing([path_2023, path_2024])

    missing_frp_rows = result.clean_df[~result.clean_df["frp_valid"]]
    assert len(missing_frp_rows) == 1
    assert pd.isna(missing_frp_rows.iloc[0]["frp"])


def test_pipeline_preserves_confidence_values_without_remapping(two_year_dataset: tuple[Path, Path]) -> None:
    path_2023, path_2024 = two_year_dataset

    result = run_firms_preprocessing([path_2023, path_2024])

    assert set(result.clean_df["confidence"].unique()) <= {"n", "l", "h"}
    assert "h" in set(result.clean_df["confidence"].unique())
    assert "l" in set(result.clean_df["confidence"].unique())


def test_pipeline_removes_exact_duplicates(two_year_dataset: tuple[Path, Path]) -> None:
    path_2023, path_2024 = two_year_dataset

    result = run_firms_preprocessing([path_2023, path_2024])

    assert result.report["duplicate_detection"]["exact_duplicate_count"] == 1


def test_pipeline_does_not_modify_raw_input_files(two_year_dataset: tuple[Path, Path]) -> None:
    path_2023, path_2024 = two_year_dataset
    original_2023 = path_2023.read_bytes()
    original_2024 = path_2024.read_bytes()

    run_firms_preprocessing([path_2023, path_2024])

    assert path_2023.read_bytes() == original_2023
    assert path_2024.read_bytes() == original_2024


def test_save_clean_dataset_writes_csv(tmp_path: Path, two_year_dataset: tuple[Path, Path]) -> None:
    path_2023, path_2024 = two_year_dataset
    result = run_firms_preprocessing([path_2023, path_2024])

    output_path = tmp_path / "out" / "clean.csv"
    save_clean_dataset(result.clean_df, output_path)

    assert output_path.exists()
    reloaded = pd.read_csv(output_path)
    assert len(reloaded) == len(result.clean_df)
