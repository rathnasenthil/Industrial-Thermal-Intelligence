"""
Coordinate validation for FIRMS thermal detections.

A detection is only spatially usable downstream (ST-DBSCAN event
formation, OSM context lookup, etc.) if it has a numeric latitude in
[-90, 90] and a numeric longitude in [-180, 180]. This module identifies
(but does not silently drop) records that fail that check.
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

LATITUDE_MIN = -90.0
LATITUDE_MAX = 90.0
LONGITUDE_MIN = -180.0
LONGITUDE_MAX = 180.0


class CoordinateValidationResult(NamedTuple):
    """Result of validating the ``latitude``/``longitude`` columns.

    Attributes:
        valid_mask: Boolean Series, ``True`` where both coordinates are
            numeric and within their valid ranges.
        stats: Summary counts describing why records were invalid.
    """

    valid_mask: pd.Series
    stats: dict[str, int]


def validate_coordinates(df: pd.DataFrame) -> CoordinateValidationResult:
    """Validate the ``latitude`` and ``longitude`` columns of ``df``.

    ``df['latitude']`` and ``df['longitude']`` are expected to already be
    numeric (see ``src.preprocessing.numeric_fields.convert_numeric_columns``).
    Non-numeric (NaN) coordinates are treated as invalid/missing rather than
    out-of-range.

    Args:
        df: DataFrame containing numeric ``latitude`` and ``longitude``
            columns.

    Returns:
        A :class:`CoordinateValidationResult` with a boolean valid mask and
        a stats dict containing:

        * ``missing_count``: coordinates that are NaN (could not be
          parsed as numbers).
        * ``out_of_range_count``: coordinates that are numeric but fall
          outside the valid lat/lon bounds.
        * ``invalid_count``: total invalid rows (``missing_count`` +
          ``out_of_range_count``); this is the number of rows that would
          be excluded from a spatially-usable dataset.
    """
    lat = df["latitude"]
    lon = df["longitude"]

    lat_missing = lat.isna()
    lon_missing = lon.isna()
    missing_mask = lat_missing | lon_missing

    lat_in_range = lat.between(LATITUDE_MIN, LATITUDE_MAX)
    lon_in_range = lon.between(LONGITUDE_MIN, LONGITUDE_MAX)
    # `.between` on NaN yields False, so combine explicitly to keep the
    # "missing" vs "out of range" distinction clean.
    out_of_range_mask = (~missing_mask) & ~(lat_in_range & lon_in_range)

    valid_mask = (~missing_mask) & lat_in_range & lon_in_range

    stats = {
        "missing_count": int(missing_mask.sum()),
        "out_of_range_count": int(out_of_range_mask.sum()),
        "invalid_count": int((~valid_mask).sum()),
    }
    return CoordinateValidationResult(valid_mask=valid_mask, stats=stats)
