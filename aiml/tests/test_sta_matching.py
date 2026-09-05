"""Tests for STA spatial/temporal matching and ranking."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.sta_evidence.config import (
    LAYER_DETECTION,
    LAYER_MASK,
    NO_STA_ASSOCIATION,
    STA_AMBIGUOUS,
    STA_ASSOCIATED,
    STA_INTERSECTS_EVENT,
    STA_NEAR_EVENT,
    STAConfig,
    TEMPORAL_NOT_APPLICABLE,
    TEMPORAL_SAME_PERIOD,
)
from src.sta_evidence.sta_matching import (
    build_event_geometries,
    classify_temporal_relation,
    find_sta_candidate_pairs,
)
from src.sta_evidence.sta_normalization import canonical_to_geodataframe, normalize_sta_geodataframe
from src.sta_evidence.sta_ranking import rank_sta_candidates, select_primary_sta_association
from tests.fixtures.sta.make_fixtures import (
    load_det_as_gdf,
    load_mask_as_gdf,
    make_synthetic_events,
    write_synthetic_sta_detections_geojson,
    write_synthetic_sta_mask_geojson,
)


@pytest.fixture()
def sta_valid(tmp_path: Path):
    mask = write_synthetic_sta_mask_geojson(tmp_path / "mask.geojson")
    det = write_synthetic_sta_detections_geojson(tmp_path / "det.geojson")
    import pandas as pd
    import geopandas as gpd

    combined = pd.concat([load_mask_as_gdf(mask), load_det_as_gdf(det)], ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")
    canonical, _ = normalize_sta_geodataframe(combined, STAConfig())
    return canonical_to_geodataframe(canonical)


def test_point_inside_polygon_intersects(sta_valid) -> None:
    events = make_synthetic_events()
    events_gdf = build_event_geometries(events)
    pairs = find_sta_candidate_pairs(events_gdf, sta_valid, STAConfig(association_radius_km=1.0))
    inside = pairs.loc[pairs["event_id"] == "EVT_INSIDE"]
    assert not inside.empty
    assert (inside["relationship"] == STA_INTERSECTS_EVENT).any()


def test_near_and_none(sta_valid) -> None:
    events = make_synthetic_events()
    events_gdf = build_event_geometries(events)
    pairs = find_sta_candidate_pairs(events_gdf, sta_valid, STAConfig(association_radius_km=2.0))
    assert "EVT_NONE" not in set(pairs["event_id"]) or pairs.loc[pairs["event_id"] == "EVT_NONE"].empty
    # EVT_NEAR should appear with NEAR or INTERSECTS depending on buffer
    near_rows = pairs.loc[pairs["event_id"] == "EVT_NEAR"]
    assert not near_rows.empty
    assert near_rows["relationship"].isin([STA_NEAR_EVENT, STA_INTERSECTS_EVENT]).all()


def test_multipolygon_supported(tmp_path: Path) -> None:
    from shapely.geometry import MultiPolygon, Polygon
    import geopandas as gpd

    mp = MultiPolygon(
        [
            Polygon([(77.0, 28.0), (77.02, 28.0), (77.02, 28.02), (77.0, 28.02)]),
        ]
    )
    sta = gpd.GeoDataFrame(
        {"id": ["MP1"], "_sta_layer_type": [LAYER_MASK], "observation_datetime": [None]},
        geometry=[mp],
        crs="EPSG:4326",
    )
    canonical, stats = normalize_sta_geodataframe(sta, STAConfig())
    assert stats["records_valid"] == 1
    sta_gdf = canonical_to_geodataframe(canonical)
    events = make_synthetic_events().iloc[[0]]
    pairs = find_sta_candidate_pairs(build_event_geometries(events), sta_gdf, STAConfig())
    assert not pairs.empty


def test_temporal_mask_not_applicable() -> None:
    assert (
        classify_temporal_relation(
            layer_type=LAYER_MASK,
            observation_datetime=None,
            event_start="2023-01-01T00:00:00+00:00",
            event_end="2023-01-02T00:00:00+00:00",
            near_hours=24,
        )
        == TEMPORAL_NOT_APPLICABLE
    )


def test_temporal_detection_same_period() -> None:
    assert (
        classify_temporal_relation(
            layer_type=LAYER_DETECTION,
            observation_datetime="2023-06-15T10:00:00+00:00",
            event_start="2023-06-15T09:00:00+00:00",
            event_end="2023-06-15T11:00:00+00:00",
            near_hours=24,
        )
        == TEMPORAL_SAME_PERIOD
    )


def test_deterministic_ranking_and_ambiguity() -> None:
    candidates = pd.DataFrame(
        [
            {
                "event_id": "E1",
                "sta_id": "sta_b",
                "sta_layer_type": LAYER_MASK,
                "relationship": STA_NEAR_EVENT,
                "distance_km": 0.50,
                "intersection_area_m2": None,
                "sta_temporal_relation": TEMPORAL_NOT_APPLICABLE,
            },
            {
                "event_id": "E1",
                "sta_id": "sta_a",
                "sta_layer_type": LAYER_MASK,
                "relationship": STA_NEAR_EVENT,
                "distance_km": 0.51,
                "intersection_area_m2": None,
                "sta_temporal_relation": TEMPORAL_NOT_APPLICABLE,
            },
        ]
    )
    config = STAConfig(ambiguity_distance_tolerance_km=0.1)
    ranked = rank_sta_candidates(candidates, config)
    assert list(ranked.sort_values("candidate_rank")["sta_id"]) == ["sta_b", "sta_a"]
    assoc = select_primary_sta_association(pd.Series(["E1"]), ranked, config)
    row = assoc.iloc[0]
    assert row["sta_association_status"] == STA_AMBIGUOUS
    assert pd.isna(row["primary_sta_id"]) or row["primary_sta_id"] is None


def test_single_clear_winner_associated() -> None:
    candidates = pd.DataFrame(
        [
            {
                "event_id": "E1",
                "sta_id": "sta_close",
                "sta_layer_type": LAYER_MASK,
                "relationship": STA_INTERSECTS_EVENT,
                "distance_km": 0.0,
                "intersection_area_m2": 100.0,
                "sta_temporal_relation": TEMPORAL_NOT_APPLICABLE,
            },
            {
                "event_id": "E1",
                "sta_id": "sta_far",
                "sta_layer_type": LAYER_DETECTION,
                "relationship": STA_NEAR_EVENT,
                "distance_km": 0.9,
                "intersection_area_m2": None,
                "sta_temporal_relation": TEMPORAL_SAME_PERIOD,
            },
        ]
    )
    ranked = rank_sta_candidates(candidates, STAConfig())
    assoc = select_primary_sta_association(pd.Series(["E1", "E2"]), ranked, STAConfig())
    e1 = assoc.loc[assoc["event_id"] == "E1"].iloc[0]
    e2 = assoc.loc[assoc["event_id"] == "E2"].iloc[0]
    assert e1["sta_association_status"] == STA_ASSOCIATED
    assert e1["primary_sta_id"] == "sta_close"
    assert e2["sta_association_status"] == NO_STA_ASSOCIATION
