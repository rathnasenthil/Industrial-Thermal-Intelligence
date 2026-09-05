"""
Integration tests for GIFT Stage I.2 (`association_pipeline`).

Covers the numbered edge cases from the Stage I.2 task spec using tiny
synthetic geometries -- never the production 179,740-event / 112,956-
facility dataset.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon

from src.event_formation.geometry import compute_event_geometry
from src.infrastructure.association_config import AssociationConfig
from src.infrastructure.association_pipeline import load_events, run_facility_association, save_outputs
from src.infrastructure.facility_association import (
    AMBIGUOUS,
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_NONE,
    NO_FACILITY_ASSOCIATION,
)
from src.infrastructure.association_geometry import INTERSECTS_FACILITY, NEAR_FACILITY, WITHIN_FACILITY

_FORBIDDEN_SUBSTRINGS = ("source_class", "industrial_fire", "wildfire", "agricultural_fire", "fire_confidence", "source_confidence")


def _event(event_id: str, lonlat_points: list[tuple[float, float]], **extra) -> dict:
    lats = np.array([p[1] for p in lonlat_points])
    lons = np.array([p[0] for p in lonlat_points])
    geom = compute_event_geometry(lats, lons)
    row = {
        "event_id": event_id,
        "detection_count": len(lonlat_points),
        "centroid_latitude": geom.centroid_latitude,
        "centroid_longitude": geom.centroid_longitude,
        "centroid_wkt": geom.centroid_wkt,
        "footprint_wkt": geom.footprint_wkt,
        "peak_frp": 5.0,
        "persistence_label": "SHORT_LIVED",
    }
    row.update(extra)
    return row


def _facility(facility_id: str, geometry_type: str, geometry, **extra) -> dict:
    row = {
        "facility_id": facility_id,
        "facility_name": extra.pop("facility_name", f"Name for {facility_id}"),
        "facility_type": extra.pop("facility_type", "OTHER_INDUSTRIAL"),
        "geometry_type": geometry_type,
        "geometry": geometry,
    }
    row.update(extra)
    return row


@pytest.fixture()
def synthetic_facilities_path(tmp_path: Path) -> Path:
    rows = [
        # Case 1/8: polygon facility for containment.
        _facility(
            "osm_way_refinery",
            "Polygon",
            Polygon([(70.000, 15.000), (70.010, 15.000), (70.010, 15.010), (70.000, 15.010)]),
            facility_type="REFINERY",
            facility_name="Test Refinery",
        ),
        # Case 3/7: point facility for near-distance association.
        _facility("osm_node_plant", "Point", Point(75.000, 15.000), facility_type="POWER_PLANT", facility_name="Test Power Plant"),
        # Case 5: two close-together mines for ambiguity.
        _facility("osm_node_mine_a", "Point", Point(78.0000, 15.0000), facility_type="MINE"),
        _facility("osm_node_mine_b", "Point", Point(78.0005, 15.0005), facility_type="MINE"),
        # Case 9: multipolygon facility.
        _facility(
            "osm_relation_area",
            "MultiPolygon",
            MultiPolygon(
                [
                    Polygon([(85.000, 15.000), (85.010, 15.000), (85.010, 15.010), (85.000, 15.010)]),
                    Polygon([(85.020, 15.020), (85.030, 15.020), (85.030, 15.030), (85.020, 15.030)]),
                ]
            ),
            facility_type="INDUSTRIAL_AREA",
        ),
        # Case 11: two facilities at identical distance from an upcoming event.
        _facility("osm_node_tie_z", "Point", Point(90.001, 15.000), facility_type="MINE"),
        _facility("osm_node_tie_a", "Point", Point(89.999, 15.000), facility_type="MINE"),
    ]
    gdf = gpd.GeoDataFrame(
        {k: [r[k] for r in rows] for k in ("facility_id", "facility_name", "facility_type", "geometry_type")},
        geometry=[r["geometry"] for r in rows],
        crs="EPSG:4326",
    )
    path = tmp_path / "facilities.geojson"
    gdf.to_file(path, driver="GeoJSON")
    return path


@pytest.fixture()
def synthetic_events_df() -> pd.DataFrame:
    rows = [
        # Case 1: fully inside the refinery polygon.
        _event("EVT_WITHIN", [(70.005, 15.005), (70.006, 15.006)]),
        # Case 2: triangle footprint crossing the refinery boundary, centroid outside.
        _event("EVT_INTERSECTS", [(70.005, 15.005), (70.020, 15.020), (70.020, 15.005)]),
        # Case 3/7: ~1 km from the point power plant.
        _event("EVT_NEAR", [(75.009, 15.000), (75.010, 15.000)]),
        # Case 4: far from everything (> 5 km radius).
        _event("EVT_NONE", [(120.0, 15.0), (120.001, 15.001)]),
        # Case 5: equidistant-ish from two very close mines -> ambiguous.
        _event("EVT_AMBIGUOUS", [(78.00025, 15.00075), (78.00026, 15.00076)]),
        # Case 9: inside the second ring of the multipolygon facility.
        _event("EVT_MULTIPOLYGON", [(85.025, 15.025), (85.026, 15.026)]),
        # Case 11: exactly between two facilities at identical distance
        # (both detections at the same point so the centroid lands exactly
        # on the midpoint between osm_node_tie_a and osm_node_tie_z).
        _event("EVT_TIE", [(90.000, 15.000), (90.000, 15.000)]),
        # Case 12: no facilities anywhere nearby (paired with an empty facility layer test separately).
        _event("EVT_ISOLATED", [(150.0, -5.0), (150.001, -5.001)]),
    ]
    return pd.DataFrame(rows)


def test_every_event_is_preserved_and_row_count_matches(synthetic_events_df: pd.DataFrame, synthetic_facilities_path: Path) -> None:
    result = run_facility_association(synthetic_events_df, synthetic_facilities_path, AssociationConfig())
    assert len(result.events_df) == len(synthetic_events_df)
    assert set(result.events_df["event_id"]) == set(synthetic_events_df["event_id"])
    assert result.events_df["event_id"].is_unique


def test_all_original_event_columns_are_preserved(synthetic_events_df: pd.DataFrame, synthetic_facilities_path: Path) -> None:
    result = run_facility_association(synthetic_events_df, synthetic_facilities_path, AssociationConfig())
    for col in synthetic_events_df.columns:
        assert col in result.events_df.columns
    pd.testing.assert_series_equal(
        result.events_df.sort_values("event_id")["peak_frp"].reset_index(drop=True),
        synthetic_events_df.sort_values("event_id")["peak_frp"].reset_index(drop=True),
        check_names=False,
    )


def test_case1_within_facility() -> None:
    pass  # covered via parametrized case test below


@pytest.mark.parametrize(
    ("event_id", "expected_method", "expected_confidence"),
    [
        ("EVT_WITHIN", WITHIN_FACILITY, CONFIDENCE_HIGH),
        ("EVT_INTERSECTS", INTERSECTS_FACILITY, CONFIDENCE_HIGH),
        ("EVT_NEAR", NEAR_FACILITY, CONFIDENCE_MEDIUM),
        ("EVT_NONE", NO_FACILITY_ASSOCIATION, CONFIDENCE_NONE),
        ("EVT_AMBIGUOUS", AMBIGUOUS, CONFIDENCE_LOW),
        ("EVT_MULTIPOLYGON", WITHIN_FACILITY, CONFIDENCE_HIGH),
        ("EVT_ISOLATED", NO_FACILITY_ASSOCIATION, CONFIDENCE_NONE),
    ],
)
def test_expected_association_cases(
    synthetic_events_df: pd.DataFrame,
    synthetic_facilities_path: Path,
    event_id: str,
    expected_method: str,
    expected_confidence: str,
) -> None:
    result = run_facility_association(synthetic_events_df, synthetic_facilities_path, AssociationConfig())
    row = result.events_df.loc[result.events_df["event_id"] == event_id].iloc[0]
    assert row["facility_association_method"] == expected_method
    assert row["facility_attribution_confidence"] == expected_confidence


def test_case11_identical_distance_tie_is_deterministic(synthetic_events_df: pd.DataFrame, synthetic_facilities_path: Path) -> None:
    # Two facilities at genuinely identical distance is exactly the
    # scenario the ambiguity rule exists for: neither is blindly picked,
    # and the (non-)selection is byte-for-byte identical across runs.
    r1 = run_facility_association(synthetic_events_df, synthetic_facilities_path, AssociationConfig())
    r2 = run_facility_association(synthetic_events_df, synthetic_facilities_path, AssociationConfig())
    row1 = r1.events_df.loc[r1.events_df["event_id"] == "EVT_TIE"].iloc[0]
    row2 = r2.events_df.loc[r2.events_df["event_id"] == "EVT_TIE"].iloc[0]
    assert row1["facility_association_method"] == AMBIGUOUS
    assert pd.isna(row1["facility_id"])
    assert row1["candidate_facility_ids"] == row2["candidate_facility_ids"]
    assert row1["candidate_facility_ids"] == "osm_node_tie_a,osm_node_tie_z"


def test_case6_no_source_classification_fields_anywhere(synthetic_events_df: pd.DataFrame, synthetic_facilities_path: Path) -> None:
    result = run_facility_association(synthetic_events_df, synthetic_facilities_path, AssociationConfig())
    all_columns = " ".join(result.events_df.columns).lower() + " " + " ".join(result.candidates_df.columns).lower()
    for forbidden in _FORBIDDEN_SUBSTRINGS:
        assert forbidden not in all_columns


def test_case12_empty_facility_layer_leaves_all_events_unassociated(tmp_path: Path, synthetic_events_df: pd.DataFrame) -> None:
    # An empty GeoJSON FeatureCollection has no features to infer a schema
    # from, so a CSV (which always carries its header row) is used here to
    # exercise the "zero facilities available" case cleanly.
    empty_df = pd.DataFrame(columns=["facility_id", "facility_name", "facility_type", "geometry_type", "geometry_wkt"])
    path = tmp_path / "empty_facilities.csv"
    empty_df.to_csv(path, index=False)

    result = run_facility_association(synthetic_events_df, path, AssociationConfig())
    assert len(result.events_df) == len(synthetic_events_df)
    assert (result.events_df["facility_association_method"] == NO_FACILITY_ASSOCIATION).all()
    assert (result.events_df["candidate_facility_count"] == 0).all()


def test_candidates_file_references_are_valid(synthetic_events_df: pd.DataFrame, synthetic_facilities_path: Path) -> None:
    result = run_facility_association(synthetic_events_df, synthetic_facilities_path, AssociationConfig())
    valid_event_ids = set(result.events_df["event_id"])
    facilities_gdf = gpd.read_file(synthetic_facilities_path)
    valid_facility_ids = set(facilities_gdf["facility_id"])

    assert set(result.candidates_df["event_id"]).issubset(valid_event_ids)
    assert set(result.candidates_df["facility_id"]).issubset(valid_facility_ids)


def test_distance_km_always_non_negative(synthetic_events_df: pd.DataFrame, synthetic_facilities_path: Path) -> None:
    result = run_facility_association(synthetic_events_df, synthetic_facilities_path, AssociationConfig())
    assert (result.candidates_df["distance_km"] >= 0).all()
    valid_distances = result.events_df["facility_distance_km"].dropna()
    assert (valid_distances >= 0).all()


def test_candidate_rank_is_deterministic_across_runs(synthetic_events_df: pd.DataFrame, synthetic_facilities_path: Path) -> None:
    r1 = run_facility_association(synthetic_events_df, synthetic_facilities_path, AssociationConfig())
    r2 = run_facility_association(synthetic_events_df, synthetic_facilities_path, AssociationConfig())
    c1 = r1.candidates_df.sort_values(["event_id", "facility_id"]).reset_index(drop=True)
    c2 = r2.candidates_df.sort_values(["event_id", "facility_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(c1, c2)


def test_full_pipeline_is_deterministic_across_repeated_runs(synthetic_events_df: pd.DataFrame, synthetic_facilities_path: Path) -> None:
    r1 = run_facility_association(synthetic_events_df, synthetic_facilities_path, AssociationConfig())
    r2 = run_facility_association(synthetic_events_df, synthetic_facilities_path, AssociationConfig())
    pd.testing.assert_frame_equal(
        r1.events_df.sort_values("event_id").reset_index(drop=True),
        r2.events_df.sort_values("event_id").reset_index(drop=True),
    )
    assert r1.report["association_results"] == r2.report["association_results"]


def test_max_candidates_per_event_caps_candidates_file(synthetic_events_df: pd.DataFrame, synthetic_facilities_path: Path) -> None:
    config = AssociationConfig(max_candidates_per_event=1)
    result = run_facility_association(synthetic_events_df, synthetic_facilities_path, config)
    per_event_counts = result.candidates_df.groupby("event_id").size()
    assert (per_event_counts <= 1).all()
    # The cap must not affect the main selection logic itself.
    ambiguous_row = result.events_df.loc[result.events_df["event_id"] == "EVT_AMBIGUOUS"].iloc[0]
    assert ambiguous_row["candidate_facility_count"] == 2  # unaffected by the output-file cap


def test_save_outputs_writes_expected_files_and_no_string_nan(tmp_path: Path, synthetic_events_df: pd.DataFrame, synthetic_facilities_path: Path) -> None:
    result = run_facility_association(synthetic_events_df, synthetic_facilities_path, AssociationConfig())
    events_out = tmp_path / "thermal_events_with_facility_association.csv"
    candidates_out = tmp_path / "thermal_event_facility_candidates.csv"
    save_outputs(result, events_out, candidates_out)

    assert events_out.exists()
    assert candidates_out.exists()

    reloaded = pd.read_csv(events_out)
    assert len(reloaded) == len(synthetic_events_df)
    # No-association rows must reload as real NaN, never the literal string "nan".
    none_rows = reloaded.loc[reloaded["facility_association_method"] == NO_FACILITY_ASSOCIATION]
    assert none_rows["facility_id"].isna().all()
    assert not (reloaded["facility_id"].astype(str) == "nan").any() or reloaded["facility_id"].isna().any()


def test_load_events_validates_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad_events.csv"
    pd.DataFrame({"event_id": ["E1"]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_events(path)


def test_load_events_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_events(tmp_path / "nope.csv")


def test_report_contains_required_sections(synthetic_events_df: pd.DataFrame, synthetic_facilities_path: Path) -> None:
    result = run_facility_association(synthetic_events_df, synthetic_facilities_path, AssociationConfig())
    report = result.report
    for key in (
        "input",
        "association_results",
        "facility_type_counts",
        "confidence_counts",
        "distance_statistics_km",
        "candidate_statistics",
        "performance",
        "configuration",
        "limitations",
    ):
        assert key in report
    assert report["input"]["event_count"] == len(synthetic_events_df)
    assert len(report["limitations"]) > 0
