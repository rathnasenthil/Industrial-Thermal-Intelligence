"""
Preprocessing report assembly for the FIRMS data-quality pipeline.

Builds a single JSON-serializable summary of everything the pipeline did
so results are auditable: row counts at each stage, missing-value counts,
invalid coordinate/timestamp/FRP counts, duplicate counts, and the final
dataset's date range and geographic bounds. Nothing here performs
imputation or cleaning — it only reports.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _to_jsonable(value: Any) -> Any:
    """Recursively convert numpy/pandas scalars to native Python types."""
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if value is pd.NaT:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def build_missing_value_summary(df: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    """Count missing (NaN/NA/NaT) values per column.

    Args:
        df: DataFrame to inspect.
        columns: Columns to include in the summary.

    Returns:
        Mapping of column name to missing-value count.
    """
    return {col: int(df[col].isna().sum()) for col in columns if col in df.columns}


def build_geographic_bounds(df: pd.DataFrame) -> dict[str, float | None]:
    """Compute the lat/lon bounding box of the (valid) rows in ``df``."""
    if df.empty or df["latitude"].isna().all():
        return {"lat_min": None, "lat_max": None, "lon_min": None, "lon_max": None}
    return {
        "lat_min": float(df["latitude"].min()),
        "lat_max": float(df["latitude"].max()),
        "lon_min": float(df["longitude"].min()),
        "lon_max": float(df["longitude"].max()),
    }


def build_date_range(df: pd.DataFrame) -> dict[str, str | None]:
    """Compute the min/max ``acq_datetime`` (UTC, ISO 8601) in ``df``."""
    if df.empty or df["acq_datetime"].isna().all():
        return {"start_utc": None, "end_utc": None}
    return {
        "start_utc": df["acq_datetime"].min().isoformat(),
        "end_utc": df["acq_datetime"].max().isoformat(),
    }


def build_preprocessing_report(
    *,
    input_file_stats: list[dict[str, Any]],
    total_input_rows: int,
    combined_rows: int,
    numeric_conversion_stats: dict[str, dict[str, int]],
    coordinate_stats: dict[str, int],
    frp_stats: dict[str, int],
    timestamp_stats: dict[str, int],
    detected_date_format: str,
    duplicate_stats: dict[str, int],
    deduplication_strategy: str,
    missing_values_by_column: dict[str, int],
    final_df: pd.DataFrame,
    invalid_rows: int,
    valid_rows: int,
    confidence_value_counts: dict[str, int],
    daynight_value_counts: dict[str, int],
) -> dict[str, Any]:
    """Assemble the full FIRMS preprocessing report as a JSON-safe dict.

    Args:
        input_file_stats: Per-input-file metadata (path, source_file tag,
            row count).
        total_input_rows: Sum of raw row counts across all input files.
        combined_rows: Row count after concatenating input files (should
            equal ``total_input_rows``).
        numeric_conversion_stats: Output of
            ``numeric_fields.convert_numeric_columns`` stats.
        coordinate_stats: Output of ``coordinates.validate_coordinates``
            stats.
        frp_stats: Output of ``numeric_fields.validate_frp`` stats.
        timestamp_stats: Output of ``timestamps.build_acq_datetime`` stats.
        detected_date_format: The auto-detected ``acq_date`` format.
        duplicate_stats: Output of ``duplicates.detect_exact_duplicates``
            stats.
        deduplication_strategy: Human-readable explanation of the
            deduplication approach.
        missing_values_by_column: Missing-value counts per column on the
            combined (pre-drop) dataset.
        final_df: The final cleaned DataFrame that will be written to
            disk.
        invalid_rows: Number of rows excluded from ``final_df`` due to
            invalid coordinates, invalid timestamps, or exact duplication.
        valid_rows: Number of rows retained in ``final_df``.
        confidence_value_counts: Value counts of the (normalized)
            ``confidence`` column in the final dataset.
        daynight_value_counts: Value counts of the (normalized)
            ``daynight`` column in the final dataset.

    Returns:
        A JSON-serializable dict, ready for ``json.dump``.
    """
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_stage": "FIRMS ingestion + data-quality preprocessing (GIFT stage: pre-G)",
        "input_files": input_file_stats,
        "row_counts": {
            "total_input_rows": total_input_rows,
            "combined_rows_before_cleaning": combined_rows,
            "invalid_rows_removed": invalid_rows,
            "final_processed_rows": valid_rows,
        },
        "coordinate_validation": coordinate_stats,
        "timestamp_validation": {
            **timestamp_stats,
            "detected_acq_date_format": detected_date_format,
        },
        "numeric_field_conversion": numeric_conversion_stats,
        "frp_validation": frp_stats,
        "duplicate_detection": {
            **duplicate_stats,
            "deduplication_strategy": deduplication_strategy,
        },
        "missing_values_by_column": missing_values_by_column,
        "confidence_value_counts": confidence_value_counts,
        "daynight_value_counts": daynight_value_counts,
        "date_range_utc": build_date_range(final_df),
        "geographic_bounds": build_geographic_bounds(final_df),
        "notes": [
            "This report covers FIRMS ingestion and data-quality preprocessing "
            "only: no event formation (ST-DBSCAN), OSM/STA/Sentinel context, "
            "facility fingerprinting, anomaly detection, or risk scoring is "
            "performed at this stage.",
            "Low FIRMS confidence ('l') is not treated as invalid and is not "
            "removed. Persistent thermal detections are preserved for the "
            "later Facility Fingerprinting stage.",
            "Missing/invalid FRP values are preserved as missing (NaN), never "
            "replaced with zero.",
        ],
    }
    return _to_jsonable(report)


def save_report(report: dict[str, Any], path: str | Path) -> None:
    """Write the report dict to disk as pretty-printed JSON.

    Args:
        report: JSON-serializable report dict (see
            :func:`build_preprocessing_report`).
        path: Output path for the JSON report.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=False)
