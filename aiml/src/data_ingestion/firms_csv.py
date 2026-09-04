"""
NASA FIRMS VIIRS CSV ingestion.

This module loads *already downloaded* FIRMS VIIRS active-fire CSV exports
(e.g. from the FIRMS archive download tool) from disk, validates that they
contain the expected columns, and combines multiple files (e.g. one per
year) into a single DataFrame while preserving provenance via a
``source_file`` column.

This module intentionally does NOT talk to the FIRMS API (see
``firms.py`` for the future live-fetch placeholder) and does NOT perform
any data-quality cleaning — that lives in ``src.preprocessing``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

# Columns every FIRMS VIIRS CSV export used by this project must contain.
# Verified against the real NOAA-20 (N20) VIIRS archive exports in
# aiml/data/raw/ (columns are read as strings; type conversion happens in
# src.preprocessing.numeric_fields).
REQUIRED_COLUMNS: tuple[str, ...] = (
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
)


class FirmsSchemaError(ValueError):
    """Raised when a FIRMS CSV file does not contain the expected columns."""


def _validate_schema(columns: Iterable[str], path: Path) -> None:
    """Ensure ``columns`` contains every entry in :data:`REQUIRED_COLUMNS`.

    Args:
        columns: Column names found in the loaded CSV.
        path: Path of the CSV file being validated (used in the error
            message so the user knows exactly which file is malformed).

    Raises:
        FirmsSchemaError: If one or more required columns are missing.
    """
    found = set(columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in found]
    if missing:
        raise FirmsSchemaError(
            f"FIRMS CSV '{path}' is missing required column(s): {missing}. "
            f"Expected columns: {list(REQUIRED_COLUMNS)}. "
            f"Found columns: {sorted(found)}."
        )


def load_firms_csv(path: str | Path) -> pd.DataFrame:
    """Load a single raw FIRMS VIIRS CSV file without mutating it.

    All columns are read as strings (``dtype=str``) so that no implicit
    numeric/date parsing happens at load time — explicit, auditable
    conversion is done later in ``src.preprocessing``. A ``source_file``
    column (the CSV's file name) is added so downstream stages can always
    trace a row back to its origin file (e.g. 2023 vs 2024).

    Args:
        path: Path to a FIRMS VIIRS CSV export.

    Returns:
        A DataFrame with the original columns (as strings) plus
        ``source_file``. The file on disk is never modified.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        FirmsSchemaError: If required columns are missing.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"FIRMS CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=True)
    _validate_schema(df.columns, csv_path)

    df = df.copy()
    df["source_file"] = csv_path.name
    return df


def load_firms_csv_files(paths: Sequence[str | Path]) -> pd.DataFrame:
    """Load and combine multiple FIRMS VIIRS CSV files.

    Each file is validated and tagged with its own ``source_file`` value
    (see :func:`load_firms_csv`) before being concatenated. Row order
    within each file is preserved; files are concatenated in the order
    given.

    Args:
        paths: Paths to one or more FIRMS VIIRS CSV files (e.g. the 2023
            and 2024 archive exports).

    Returns:
        A single combined DataFrame containing all rows from all files.

    Raises:
        ValueError: If ``paths`` is empty.
        FileNotFoundError: If any path does not exist.
        FirmsSchemaError: If any file is missing required columns.
    """
    if not paths:
        raise ValueError("At least one FIRMS CSV path must be provided.")

    frames = [load_firms_csv(p) for p in paths]
    combined = pd.concat(frames, ignore_index=True)
    return combined
