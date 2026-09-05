"""Tests for src.infrastructure.facility_validation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.infrastructure.facility_schema import CANONICAL_COLUMNS
from src.infrastructure.facility_validation import (
    build_rejection_reasons,
    detect_duplicate_facility_ids,
    validate_facilities,
)


def _facility_row(**overrides) -> dict:
    base = {c: None for c in CANONICAL_COLUMNS}
    base.update(
        {
            "facility_id": "osm_way_1",
            "facility_name": "Test",
            "facility_type": "INDUSTRIAL_AREA",
            "geometry_type": "Point",
            "latitude": 20.0,
            "longitude": 80.0,
            "geometry_wkt": "POINT (80 20)",
            "osm_id": "1",
            "osm_type": "way",
        }
    )
    base.update(overrides)
    return base


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS))


def test_valid_record_passes_all_checks() -> None:
    df = _df([_facility_row()])
    result = validate_facilities(df)
    assert result.valid_mask.iloc[0]
    assert result.stats["valid_record_count"] == 1
    assert result.stats["invalid_record_count"] == 0


def test_missing_facility_id_is_flagged() -> None:
    df = _df([_facility_row(facility_id=None)])
    result = validate_facilities(df)
    assert not result.valid_mask.iloc[0]
    assert result.stats["missing_id_count"] == 1


def test_blank_facility_id_is_flagged() -> None:
    df = _df([_facility_row(facility_id="   ")])
    result = validate_facilities(df)
    assert result.stats["missing_id_count"] == 1


def test_missing_geometry_wkt_is_invalid_geometry() -> None:
    df = _df([_facility_row(geometry_wkt=None)])
    result = validate_facilities(df)
    assert not result.valid_mask.iloc[0]
    assert result.stats["invalid_geometry_count"] == 1


def test_unsupported_geometry_type_is_invalid() -> None:
    df = _df([_facility_row(geometry_type="LineString", geometry_wkt="LINESTRING (0 0, 1 1)")])
    result = validate_facilities(df)
    assert not result.valid_mask.iloc[0]
    assert result.stats["invalid_geometry_count"] == 1


def test_polygon_and_multipolygon_geometry_types_are_valid() -> None:
    df = _df(
        [
            _facility_row(geometry_type="Polygon", geometry_wkt="POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"),
            _facility_row(
                facility_id="osm_way_2",
                geometry_type="MultiPolygon",
                geometry_wkt="MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)))",
            ),
        ]
    )
    result = validate_facilities(df)
    assert result.valid_mask.all()


def test_out_of_range_coordinates_are_invalid() -> None:
    df = _df([_facility_row(latitude=999.0, longitude=80.0)])
    result = validate_facilities(df)
    assert not result.valid_mask.iloc[0]
    assert result.stats["invalid_coordinate_count"] == 1


def test_missing_coordinates_are_invalid() -> None:
    df = _df([_facility_row(latitude=None, longitude=None)])
    result = validate_facilities(df)
    assert result.stats["invalid_coordinate_count"] == 1


def test_unknown_facility_type_is_valid_but_tracked_separately() -> None:
    """UNKNOWN is a legitimate, expected outcome -- it must not make the
    record invalid, but is still counted for visibility."""
    df = _df([_facility_row(facility_type="UNKNOWN")])
    result = validate_facilities(df)
    assert result.valid_mask.iloc[0]
    assert result.stats["unknown_type_count"] == 1
    assert result.stats["valid_record_count"] == 1


def test_unsupported_facility_type_value_is_invalid() -> None:
    df = _df([_facility_row(facility_type="NOT_A_REAL_TYPE")])
    result = validate_facilities(df)
    assert not result.valid_mask.iloc[0]
    assert result.stats["unsupported_facility_type_count"] == 1


def test_empty_dataframe_produces_zeroed_stats() -> None:
    df = _df([])
    result = validate_facilities(df)
    assert result.stats["input_records"] == 0
    assert result.stats["valid_record_count"] == 0
    assert len(result.valid_mask) == 0


def test_detect_duplicate_facility_ids_flags_repeats_keeping_first() -> None:
    df = _df([_facility_row(), _facility_row(), _facility_row(facility_id="osm_way_2")])
    result = detect_duplicate_facility_ids(df)
    assert list(result.duplicate_mask) == [False, True, False]
    assert result.stats["duplicate_facility_id_count"] == 1


def test_detect_duplicate_facility_ids_does_not_flag_nearby_distinct_ids() -> None:
    """Two geographically close but distinct OSM objects (e.g. a boundary
    polygon and a separate entrance node) must NOT be treated as
    duplicates merely because facility_id differs but coordinates are
    close."""
    df = _df(
        [
            _facility_row(facility_id="osm_way_1", latitude=20.0001, longitude=80.0001),
            _facility_row(facility_id="osm_node_2", latitude=20.0001, longitude=80.0001),
        ]
    )
    result = detect_duplicate_facility_ids(df)
    assert not result.duplicate_mask.any()
    assert result.stats["duplicate_facility_id_count"] == 0


def test_detect_duplicate_facility_ids_on_empty_dataframe() -> None:
    result = detect_duplicate_facility_ids(_df([]))
    assert result.stats["duplicate_facility_id_count"] == 0


def test_build_rejection_reasons_explains_every_rejected_record() -> None:
    df = _df(
        [
            _facility_row(),  # valid
            _facility_row(facility_id=None),  # missing id
            _facility_row(facility_id="osm_way_3", latitude=999.0),  # bad coordinate
        ]
    )
    validation = validate_facilities(df)
    duplicates = detect_duplicate_facility_ids(df)
    reasons = build_rejection_reasons(df, validation, duplicates.duplicate_mask)

    assert reasons.iloc[0] == ""
    assert "missing_facility_id" in reasons.iloc[1]
    assert "invalid_coordinates" in reasons.iloc[2]


def test_build_rejection_reasons_lists_multiple_reasons_when_applicable() -> None:
    df = _df([_facility_row(facility_id=None, latitude=999.0)])
    validation = validate_facilities(df)
    duplicates = detect_duplicate_facility_ids(df)
    reasons = build_rejection_reasons(df, validation, duplicates.duplicate_mask)

    assert "missing_facility_id" in reasons.iloc[0]
    assert "invalid_coordinates" in reasons.iloc[0]
