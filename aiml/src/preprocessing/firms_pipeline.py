"""
End-to-end FIRMS ingestion + data-quality preprocessing pipeline.

This module wires together, in order:

1. ``src.data_ingestion.firms_csv`` — load & schema-validate raw CSV(s).
2. ``numeric_fields`` — convert numeric columns, validate FRP.
3. ``coordinates`` — validate latitude/longitude.
4. ``timestamps`` — build the UTC ``acq_datetime`` and derived fields.
5. ``categorical_fields`` — normalize confidence/daynight formatting.
6. ``duplicates`` — flag exact duplicate rows.
7. ``report`` — summarize everything into a JSON report.

Explicit design choice: a row is excluded from the final cleaned dataset
only if its coordinates are invalid, its timestamp could not be
constructed, or it is an exact duplicate of an earlier row. Rows with
missing/invalid FRP or with other unparsable ancillary numeric fields
(``bright_ti4``, ``bright_ti5``, ``scan``, ``track``, ``type``,
``version``) are KEPT — those columns are just left as ``NaN`` rather than
invented, per the "do not fabricate data" requirement. Low FIRMS
confidence is never treated as invalid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple, Sequence

import pandas as pd

from src.data_ingestion.firms_csv import REQUIRED_COLUMNS, load_firms_csv_files
from src.preprocessing import report as report_module
from src.preprocessing.categorical_fields import normalize_confidence, normalize_daynight
from src.preprocessing.coordinates import validate_coordinates
from src.preprocessing.duplicates import detect_exact_duplicates
from src.preprocessing.numeric_fields import convert_numeric_columns, validate_frp
from src.preprocessing.timestamps import build_acq_datetime

# Column order for the final cleaned CSV: original FIRMS columns (schema
# order) + provenance + derived fields.
_OUTPUT_COLUMNS: tuple[str, ...] = (
    "source_file",
    *REQUIRED_COLUMNS,
    "acq_datetime",
    "hour_utc",
    "day_of_week",
    "month",
    "day_of_year",
    "frp_valid",
)


class FirmsPipelineResult(NamedTuple):
    """Result of running the full FIRMS preprocessing pipeline.

    Attributes:
        clean_df: The final cleaned, combined DataFrame.
        report: JSON-serializable preprocessing report (see
            ``src.preprocessing.report.build_preprocessing_report``).
    """

    clean_df: pd.DataFrame
    report: dict[str, Any]


def run_firms_preprocessing(input_paths: Sequence[str | Path]) -> FirmsPipelineResult:
    """Run the full FIRMS ingestion + data-quality preprocessing pipeline.

    Args:
        input_paths: Paths to one or more raw FIRMS VIIRS CSV files (e.g.
            the 2023 and 2024 archive exports). Files are never modified.

    Returns:
        A :class:`FirmsPipelineResult` with the cleaned DataFrame and the
        full preprocessing report.
    """
    input_paths = [Path(p) for p in input_paths]

    # --- 1. Ingestion -----------------------------------------------------
    raw_df = load_firms_csv_files(input_paths)
    input_file_stats = []
    for p in input_paths:
        count = int((raw_df["source_file"] == p.name).sum())
        input_file_stats.append({"path": str(p), "source_file": p.name, "row_count": count})
    total_input_rows = sum(f["row_count"] for f in input_file_stats)
    combined_rows = len(raw_df)

    # --- 2. Numeric conversion (incl. FRP validation) ---------------------
    numeric_result = convert_numeric_columns(raw_df)
    df = numeric_result.df
    frp_result = validate_frp(df)
    df["frp_valid"] = frp_result.valid_mask

    # --- 3. Coordinate validation ------------------------------------------
    coord_result = validate_coordinates(df)

    # --- 4. Timestamp construction (UTC) -----------------------------------
    ts_result = build_acq_datetime(df)
    df = ts_result.df

    # --- 5. Confidence / day-night formatting normalization ----------------
    df["confidence"] = normalize_confidence(df["confidence"])
    df["daynight"] = normalize_daynight(df["daynight"])

    # --- 6. Exact duplicate detection ---------------------------------------
    dup_result = detect_exact_duplicates(df, subset=REQUIRED_COLUMNS)

    # --- Assemble missing-value summary (on combined, pre-drop data) -------
    missing_values_by_column = report_module.build_missing_value_summary(
        df, list(REQUIRED_COLUMNS) + ["source_file", "acq_datetime"]
    )

    # --- Final row selection -------------------------------------------------
    final_valid_mask = coord_result.valid_mask & ts_result.valid_mask & (~dup_result.duplicate_mask)
    invalid_rows = int((~final_valid_mask).sum())
    valid_rows = int(final_valid_mask.sum())

    clean_df = df.loc[final_valid_mask, list(_OUTPUT_COLUMNS)].reset_index(drop=True)

    confidence_value_counts = {
        str(k): int(v) for k, v in clean_df["confidence"].value_counts(dropna=False).items()
    }
    daynight_value_counts = {
        str(k): int(v) for k, v in clean_df["daynight"].value_counts(dropna=False).items()
    }

    report = report_module.build_preprocessing_report(
        input_file_stats=input_file_stats,
        total_input_rows=total_input_rows,
        combined_rows=combined_rows,
        numeric_conversion_stats=numeric_result.stats,
        coordinate_stats=coord_result.stats,
        frp_stats=frp_result.stats,
        timestamp_stats=ts_result.stats,
        detected_date_format=ts_result.detected_date_format,
        duplicate_stats=dup_result.stats,
        deduplication_strategy=dup_result.strategy_note,
        missing_values_by_column=missing_values_by_column,
        final_df=clean_df,
        invalid_rows=invalid_rows,
        valid_rows=valid_rows,
        confidence_value_counts=confidence_value_counts,
        daynight_value_counts=daynight_value_counts,
    )

    return FirmsPipelineResult(clean_df=clean_df, report=report)


def save_clean_dataset(clean_df: pd.DataFrame, output_csv: str | Path) -> None:
    """Write the cleaned dataset to CSV.

    Args:
        clean_df: The cleaned FIRMS DataFrame.
        output_csv: Destination CSV path (parent directories are created
            if needed). This never touches the raw input files.
    """
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(out_path, index=False)
