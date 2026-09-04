"""
Numeric field validation and conversion for FIRMS thermal detections.

FIRMS CSV exports are loaded as strings (see
``src.data_ingestion.firms_csv``). This module converts the numeric fields
to proper numeric dtypes and reports, per column, how many values were
missing (blank) versus present-but-non-numeric ("invalid"). It never
invents replacement values for missing/invalid numeric data — conversion
failures become ``NaN`` and are counted, not imputed.

Fire Radiative Power (``frp``) gets a dedicated validation helper because
it needs special handling: a missing/invalid FRP must never be silently
treated as zero, and a legitimately small (but present) FRP value must
never be flagged as invalid just because it is small.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

# Fields that represent physical/radiometric or categorical-numeric
# quantities and must be numeric for any downstream analysis. `type` and
# `version` are integer codes in the FIRMS product spec; they are converted
# to numeric here but their *meaning* is intentionally left uninterpreted
# (no classification happens in this preprocessing stage).
NUMERIC_COLUMNS: tuple[str, ...] = (
    "latitude",
    "longitude",
    "bright_ti4",
    "bright_ti5",
    "scan",
    "track",
    "frp",
    "type",
    "version",
)


class NumericConversionResult(NamedTuple):
    """Result of converting the FIRMS numeric columns.

    Attributes:
        df: Copy of the input DataFrame with ``columns`` converted to
            ``float64`` (invalid/missing entries become ``NaN``).
        stats: Per-column dict with ``missing`` (blank/NaN in the source)
            and ``invalid_non_numeric`` (present but unparsable) counts.
    """

    df: pd.DataFrame
    stats: dict[str, dict[str, int]]


class FrpValidationResult(NamedTuple):
    """Result of validating the ``frp`` (Fire Radiative Power) column.

    Attributes:
        valid_mask: ``True`` where FRP is a present, non-negative number.
            Small positive values (e.g. 0.1 MW) are valid — FRP is only
            flagged invalid if it is missing/unparsable or negative
            (physically impossible), never because it is "too small".
        stats: Summary counts (see :func:`validate_frp`).
    """

    valid_mask: pd.Series
    stats: dict[str, int]


def convert_numeric_columns(
    df: pd.DataFrame, columns: tuple[str, ...] = NUMERIC_COLUMNS
) -> NumericConversionResult:
    """Convert string columns to numeric, reporting missing vs invalid entries.

    For each column in ``columns``:

    * A blank/whitespace-only or already-NaN source value is counted as
      ``missing``.
    * A non-blank source value that cannot be parsed as a number is
      counted as ``invalid_non_numeric`` (this should be rare/never for
      well-formed FIRMS exports, but must not be hidden if it happens).
    * Successfully parsed values are converted to ``float64``.

    No missing or invalid value is ever replaced with an invented number
    (e.g. zero); both remain ``NaN`` in the output.

    Args:
        df: DataFrame with the columns to convert, as loaded by
            ``load_firms_csv`` (string dtype).
        columns: Names of the columns to convert.

    Returns:
        A :class:`NumericConversionResult`.
    """
    out = df.copy()
    stats: dict[str, dict[str, int]] = {}

    for column in columns:
        raw = out[column]
        raw_str = raw.astype(str).str.strip()
        blank_mask = raw.isna() | raw_str.eq("") | raw_str.str.lower().eq("nan")

        numeric = pd.to_numeric(raw, errors="coerce")
        invalid_non_numeric_mask = numeric.isna() & ~blank_mask

        out[column] = numeric
        stats[column] = {
            "missing": int(blank_mask.sum()),
            "invalid_non_numeric": int(invalid_non_numeric_mask.sum()),
        }

    return NumericConversionResult(df=out, stats=stats)


def validate_frp(df: pd.DataFrame) -> FrpValidationResult:
    """Validate the (already-numeric) ``frp`` column.

    Args:
        df: DataFrame with a numeric ``frp`` column (see
            :func:`convert_numeric_columns`).

    Returns:
        A :class:`FrpValidationResult` whose stats contain:

        * ``missing_or_unparsable``: FRP is ``NaN`` (blank in the source
          or could not be parsed as a number).
        * ``negative``: FRP parsed to a numeric value below zero, which is
          not physically possible for Fire Radiative Power and is treated
          as invalid data rather than a real observation.
        * ``invalid_count``: total of the two categories above.
        * ``valid_low_count``: valid (present, non-negative) FRP values
          that are small (< 1 MW), reported purely for transparency so it
          is clear such values are *kept*, not filtered out.
    """
    frp = df["frp"]
    missing_mask = frp.isna()
    negative_mask = (~missing_mask) & (frp < 0)
    valid_mask = (~missing_mask) & (~negative_mask)

    valid_low_mask = valid_mask & (frp < 1.0)

    stats = {
        "missing_or_unparsable": int(missing_mask.sum()),
        "negative": int(negative_mask.sum()),
        "invalid_count": int((~valid_mask).sum()),
        "valid_low_count": int(valid_low_mask.sum()),
    }
    return FrpValidationResult(valid_mask=valid_mask, stats=stats)
