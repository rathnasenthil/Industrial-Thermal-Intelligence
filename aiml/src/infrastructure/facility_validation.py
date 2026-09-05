"""
Facility-record validation for GIFT Stage I.1.

Validates the canonical facility table produced by
`osm_normalization.normalize_osm_facilities` and flags (never silently
deletes) records with a missing id, invalid/unsupported geometry, invalid
coordinates, or an unsupported `facility_type` value. `UNKNOWN` is a
supported, expected `facility_type` value — it does NOT make a record
invalid.

Duplicate handling mirrors the deliberately conservative approach used
for FIRMS detections (`src.preprocessing.duplicates`): two nearby OSM
objects (e.g. a facility boundary polygon and a separate entrance node)
are NOT duplicates of each other merely because they are geographically
close. Only records that resolve to the *same deterministic facility_id*
(i.e. genuinely the same OSM element, or byte-identical fallback content)
are treated as duplicates.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from src.infrastructure.facility_schema import FACILITY_TYPES, SUPPORTED_GEOMETRY_TYPES, UNKNOWN
from src.preprocessing.coordinates import LATITUDE_MAX, LATITUDE_MIN, LONGITUDE_MAX, LONGITUDE_MIN

DEFAULT_DUPLICATE_STRATEGY = (
    "Facilities are only treated as duplicates when they resolve to the exact "
    "same deterministic facility_id (same OSM osm_type+osm_id, or identical "
    "fallback-id content for records without a stable OSM id). Two nearby but "
    "distinct OSM objects (e.g. a facility boundary polygon, an entrance node, "
    "and a separate building) are deliberately NOT merged just because they "
    "are geographically close -- see facility_schema.make_osm_facility_id / "
    "make_fallback_facility_id."
)


class FacilityValidationResult(NamedTuple):
    """Result of validating the canonical facility table.

    Attributes:
        valid_mask: ``True`` for rows with a present id, valid/supported
            geometry, valid coordinates, and a supported `facility_type`
            value (note: `UNKNOWN` counts as supported/valid).
        stats: Validation statistics (see module-level constants for the
            exact keys).
    """

    valid_mask: pd.Series
    stats: dict[str, int]


class DuplicateFacilityResult(NamedTuple):
    """Result of duplicate-facility-id detection.

    Attributes:
        duplicate_mask: ``True`` for rows that repeat a `facility_id`
            already seen earlier in the table (keeps the first
            occurrence).
        stats: ``{"duplicate_facility_id_count": int}``.
        strategy_note: Human-readable explanation (see
            `DEFAULT_DUPLICATE_STRATEGY`).
    """

    duplicate_mask: pd.Series
    stats: dict[str, int]
    strategy_note: str


def validate_facilities(df: pd.DataFrame) -> FacilityValidationResult:
    """Validate the canonical facility table.

    Args:
        df: Output of `osm_normalization.normalize_osm_facilities` (must
            contain `facility_schema.CANONICAL_COLUMNS`).

    Returns:
        A `FacilityValidationResult`.
    """
    n = len(df)
    if n == 0:
        empty_mask = pd.Series([], dtype=bool)
        return FacilityValidationResult(
            valid_mask=empty_mask,
            stats={
                "input_records": 0,
                "missing_id_count": 0,
                "invalid_geometry_count": 0,
                "invalid_coordinate_count": 0,
                "unsupported_facility_type_count": 0,
                "unknown_type_count": 0,
                "valid_record_count": 0,
                "invalid_record_count": 0,
            },
        )

    facility_id = df["facility_id"].astype("string")
    missing_id_mask = facility_id.isna() | (facility_id.str.strip() == "")

    geometry_type = df["geometry_type"]
    geometry_wkt = df["geometry_wkt"].astype("string")
    geometry_present = geometry_wkt.notna() & (geometry_wkt.str.strip() != "")
    geometry_type_supported = geometry_type.isin(SUPPORTED_GEOMETRY_TYPES)
    invalid_geometry_mask = (~geometry_present) | (~geometry_type_supported)

    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    coordinate_missing_mask = lat.isna() | lon.isna()
    coordinate_out_of_range_mask = (~coordinate_missing_mask) & ~(
        lat.between(LATITUDE_MIN, LATITUDE_MAX) & lon.between(LONGITUDE_MIN, LONGITUDE_MAX)
    )
    invalid_coordinate_mask = coordinate_missing_mask | coordinate_out_of_range_mask

    unsupported_type_mask = ~df["facility_type"].isin(FACILITY_TYPES)
    unknown_type_mask = df["facility_type"] == UNKNOWN

    valid_mask = (
        (~missing_id_mask) & (~invalid_geometry_mask) & (~invalid_coordinate_mask) & (~unsupported_type_mask)
    )

    stats = {
        "input_records": int(n),
        "missing_id_count": int(missing_id_mask.sum()),
        "invalid_geometry_count": int(invalid_geometry_mask.sum()),
        "invalid_coordinate_count": int(invalid_coordinate_mask.sum()),
        "unsupported_facility_type_count": int(unsupported_type_mask.sum()),
        "unknown_type_count": int(unknown_type_mask.sum()),
        "valid_record_count": int(valid_mask.sum()),
        "invalid_record_count": int((~valid_mask).sum()),
    }
    return FacilityValidationResult(valid_mask=valid_mask, stats=stats)


def detect_duplicate_facility_ids(df: pd.DataFrame) -> DuplicateFacilityResult:
    """Flag rows whose `facility_id` repeats an earlier row's.

    Args:
        df: Canonical facility table (any subset of rows, e.g. already
            filtered to valid records).

    Returns:
        A `DuplicateFacilityResult`.
    """
    if len(df) == 0:
        return DuplicateFacilityResult(
            duplicate_mask=pd.Series([], dtype=bool),
            stats={"duplicate_facility_id_count": 0},
            strategy_note=DEFAULT_DUPLICATE_STRATEGY,
        )
    duplicate_mask = df["facility_id"].duplicated(keep="first")
    stats = {"duplicate_facility_id_count": int(duplicate_mask.sum())}
    return DuplicateFacilityResult(duplicate_mask=duplicate_mask, stats=stats, strategy_note=DEFAULT_DUPLICATE_STRATEGY)


def build_rejection_reasons(
    df: pd.DataFrame, validation: FacilityValidationResult, duplicate_mask: pd.Series
) -> pd.Series:
    """Human-readable reason(s) a record was excluded from the final output.

    Every rejected record gets a non-empty explanation (never silently
    dropped without one). Records failing multiple checks list all of
    them, semicolon-separated.

    Args:
        df: Canonical facility table (same rows `validation`/
            `duplicate_mask` were computed over).
        validation: Output of `validate_facilities`.
        duplicate_mask: Output of `detect_duplicate_facility_ids`
            (aligned to `df`'s index).

    Returns:
        A string Series, empty string for rows that were not rejected.
    """
    if len(df) == 0:
        return pd.Series([], dtype=str)

    facility_id = df["facility_id"].astype("string")
    missing_id_mask = facility_id.isna() | (facility_id.str.strip() == "")

    geometry_wkt = df["geometry_wkt"].astype("string")
    geometry_present = geometry_wkt.notna() & (geometry_wkt.str.strip() != "")
    geometry_type_supported = df["geometry_type"].isin(SUPPORTED_GEOMETRY_TYPES)
    invalid_geometry_mask = (~geometry_present) | (~geometry_type_supported)

    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    coordinate_missing_mask = lat.isna() | lon.isna()
    coordinate_out_of_range_mask = (~coordinate_missing_mask) & ~(
        lat.between(LATITUDE_MIN, LATITUDE_MAX) & lon.between(LONGITUDE_MIN, LONGITUDE_MAX)
    )
    invalid_coordinate_mask = coordinate_missing_mask | coordinate_out_of_range_mask

    unsupported_type_mask = ~df["facility_type"].isin(FACILITY_TYPES)

    checks = (
        (missing_id_mask, "missing_facility_id"),
        (invalid_geometry_mask, "invalid_or_unsupported_geometry"),
        (invalid_coordinate_mask, "invalid_coordinates"),
        (unsupported_type_mask, "unsupported_facility_type"),
        (duplicate_mask.reindex(df.index, fill_value=False), "duplicate_facility_id"),
    )

    reason_lists: list[list[str]] = [[] for _ in range(len(df))]
    for mask, label in checks:
        for position in np.flatnonzero(mask.to_numpy()):
            reason_lists[position].append(label)

    return pd.Series(["; ".join(labels) for labels in reason_lists], index=df.index, dtype=str)
