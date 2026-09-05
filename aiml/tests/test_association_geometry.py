"""Tests for `src.infrastructure.association_geometry`."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from src.event_formation.geometry import compute_event_geometry
from src.infrastructure.association_geometry import (
    INDIA_EQUAL_AREA_CRS,
    INTERSECTS_FACILITY,
    NEAR_FACILITY,
    WITHIN_FACILITY,
    build_event_geometries,
    find_candidate_pairs,
    load_facilities_geodataframe,
)


def _facilities_gdf(rows: list[dict]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "facility_id": [r["facility_id"] for r in rows],
            "facility_name": [r.get("facility_name", "Facility") for r in rows],
            "facility_type": [r.get("facility_type", "OTHER_INDUSTRIAL") for r in rows],
            "geometry_type": [r["geometry_type"] for r in rows],
            "geometry": [r["geometry"] for r in rows],
        },
        crs="EPSG:4326",
    )


def _event_row(event_id: str, lonlat_points: list[tuple[float, float]]) -> dict:
    lats = np.array([p[1] for p in lonlat_points])
    lons = np.array([p[0] for p in lonlat_points])
    geom = compute_event_geometry(lats, lons)
    return {
        "event_id": event_id,
        "detection_count": len(lonlat_points),
        "centroid_wkt": geom.centroid_wkt,
        "footprint_wkt": geom.footprint_wkt,
    }


# --------------------------------------------------------------------------
# load_facilities_geodataframe
# --------------------------------------------------------------------------


def test_load_facilities_geojson_round_trip(tmp_path: Path) -> None:
    gdf = _facilities_gdf(
        [
            {"facility_id": "osm_way_1", "geometry_type": "Polygon", "geometry": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])},
            {"facility_id": "osm_node_2", "geometry_type": "Point", "geometry": Point(2, 2)},
        ]
    )
    path = tmp_path / "facilities.geojson"
    gdf.to_file(path, driver="GeoJSON")

    loaded = load_facilities_geodataframe(path)
    assert len(loaded) == 2
    assert str(loaded.crs).upper() in {"EPSG:4326", "WGS84"}
    assert set(loaded["facility_id"]) == {"osm_way_1", "osm_node_2"}


def test_load_facilities_csv(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "facility_id": ["osm_node_9"],
            "facility_name": ["Test Plant"],
            "facility_type": ["POWER_PLANT"],
            "geometry_type": ["Point"],
            "geometry_wkt": ["POINT (10 20)"],
        }
    )
    path = tmp_path / "facilities.csv"
    df.to_csv(path, index=False)

    loaded = load_facilities_geodataframe(path)
    assert len(loaded) == 1
    assert loaded.iloc[0]["facility_id"] == "osm_node_9"
    assert loaded.geometry.iloc[0].equals(Point(10, 20))


def test_load_facilities_drops_unsupported_geometry_type(tmp_path: Path) -> None:
    gdf = _facilities_gdf(
        [
            {"facility_id": "osm_way_1", "geometry_type": "Polygon", "geometry": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])},
            # A LineString slipped in despite Stage I.1's own validation --
            # I.2 must defensively drop it, not crash.
            {"facility_id": "osm_way_2", "geometry_type": "LineString", "geometry": LineString([(5, 5), (6, 6)])},
        ]
    )
    path = tmp_path / "facilities.geojson"
    gdf.to_file(path, driver="GeoJSON")

    loaded = load_facilities_geodataframe(path)
    assert list(loaded["facility_id"]) == ["osm_way_1"]


def test_load_facilities_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_facilities_geodataframe(tmp_path / "does_not_exist.geojson")


def test_load_facilities_multipolygon_supported(tmp_path: Path) -> None:
    mp = MultiPolygon(
        [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(5, 5), (6, 5), (6, 6), (5, 6)]),
        ]
    )
    gdf = _facilities_gdf([{"facility_id": "osm_relation_1", "geometry_type": "MultiPolygon", "geometry": mp}])
    path = tmp_path / "facilities.geojson"
    gdf.to_file(path, driver="GeoJSON")

    loaded = load_facilities_geodataframe(path)
    assert len(loaded) == 1
    assert loaded.geometry.iloc[0].geom_type == "MultiPolygon"


# --------------------------------------------------------------------------
# build_event_geometries
# --------------------------------------------------------------------------


def test_build_event_geometries_parses_wkt() -> None:
    events_df = pd.DataFrame([_event_row("EVT_1", [(80.0, 20.0), (80.001, 20.001)])])
    gdf = build_event_geometries(events_df)
    assert len(gdf) == 1
    assert gdf.iloc[0]["centroid_geom"].geom_type == "Point"
    assert str(gdf.crs).upper() in {"EPSG:4326", "WGS84"}


def test_build_event_geometries_missing_columns_raises() -> None:
    with pytest.raises(ValueError):
        build_event_geometries(pd.DataFrame({"event_id": ["EVT_1"]}))


def test_build_event_geometries_bad_wkt_raises() -> None:
    df = pd.DataFrame({"event_id": ["EVT_1"], "centroid_wkt": ["NOT WKT"], "footprint_wkt": ["NOT WKT"]})
    with pytest.raises(ValueError):
        build_event_geometries(df)


# --------------------------------------------------------------------------
# find_candidate_pairs
# --------------------------------------------------------------------------


def test_find_candidate_pairs_within_polygon() -> None:
    facilities = _facilities_gdf(
        [{"facility_id": "F1", "geometry_type": "Polygon", "geometry": Polygon([(10.0, 10.0), (10.01, 10.0), (10.01, 10.01), (10.0, 10.01)])}]
    )
    events_df = pd.DataFrame([_event_row("EVT_1", [(10.005, 10.005), (10.006, 10.006)])])
    events_gdf = build_event_geometries(events_df)

    pairs = find_candidate_pairs(events_gdf, facilities, association_radius_km=5.0)
    assert len(pairs) == 1
    assert pairs.iloc[0]["spatial_relation"] == WITHIN_FACILITY
    assert pairs.iloc[0]["distance_km"] == pytest.approx(0.0, abs=1e-6)


def test_find_candidate_pairs_intersects_polygon() -> None:
    facilities = _facilities_gdf(
        [{"facility_id": "F1", "geometry_type": "Polygon", "geometry": Polygon([(10.0, 10.0), (10.01, 10.0), (10.01, 10.01), (10.0, 10.01)])}]
    )
    # Triangle with one vertex inside the polygon and centroid outside it
    # -> footprint intersects the polygon, but centroid does not.
    events_df = pd.DataFrame([_event_row("EVT_1", [(10.005, 10.005), (10.020, 10.020), (10.020, 10.005)])])
    events_gdf = build_event_geometries(events_df)

    pairs = find_candidate_pairs(events_gdf, facilities, association_radius_km=5.0)
    assert len(pairs) == 1
    assert pairs.iloc[0]["spatial_relation"] == INTERSECTS_FACILITY


def test_find_candidate_pairs_near_facility() -> None:
    facilities = _facilities_gdf([{"facility_id": "F1", "geometry_type": "Point", "geometry": Point(10.0, 10.0)}])
    # ~1.5 km away at this latitude.
    events_df = pd.DataFrame([_event_row("EVT_1", [(10.013, 10.0), (10.014, 10.0)])])
    events_gdf = build_event_geometries(events_df)

    pairs = find_candidate_pairs(events_gdf, facilities, association_radius_km=5.0)
    assert len(pairs) == 1
    assert pairs.iloc[0]["spatial_relation"] == NEAR_FACILITY
    assert 0 < pairs.iloc[0]["distance_km"] < 5.0


def test_find_candidate_pairs_outside_radius_returns_no_candidates() -> None:
    facilities = _facilities_gdf([{"facility_id": "F1", "geometry_type": "Point", "geometry": Point(10.0, 10.0)}])
    events_df = pd.DataFrame([_event_row("EVT_1", [(12.0, 12.0), (12.001, 12.001)])])
    events_gdf = build_event_geometries(events_df)

    pairs = find_candidate_pairs(events_gdf, facilities, association_radius_km=5.0)
    assert len(pairs) == 0
    assert list(pairs.columns) == [
        "event_id",
        "facility_id",
        "facility_name",
        "facility_type",
        "geometry_type",
        "distance_km",
        "spatial_relation",
    ]


def test_find_candidate_pairs_multipolygon_within() -> None:
    mp = MultiPolygon(
        [
            Polygon([(10.0, 10.0), (10.01, 10.0), (10.01, 10.01), (10.0, 10.01)]),
            Polygon([(20.0, 20.0), (20.01, 20.0), (20.01, 20.01), (20.0, 20.01)]),
        ]
    )
    facilities = _facilities_gdf([{"facility_id": "F1", "geometry_type": "MultiPolygon", "geometry": mp}])
    events_df = pd.DataFrame([_event_row("EVT_1", [(10.005, 10.005), (10.006, 10.006)])])
    events_gdf = build_event_geometries(events_df)

    pairs = find_candidate_pairs(events_gdf, facilities, association_radius_km=5.0)
    assert len(pairs) == 1
    assert pairs.iloc[0]["spatial_relation"] == WITHIN_FACILITY


def test_find_candidate_pairs_empty_inputs_return_empty_frame() -> None:
    facilities = _facilities_gdf([{"facility_id": "F1", "geometry_type": "Point", "geometry": Point(10.0, 10.0)}])
    empty_events = build_event_geometries(pd.DataFrame(columns=["event_id", "centroid_wkt", "footprint_wkt"]))
    pairs = find_candidate_pairs(empty_events, facilities, association_radius_km=5.0)
    assert len(pairs) == 0

    events_df = pd.DataFrame([_event_row("EVT_1", [(10.0, 10.0), (10.0, 10.0)])])
    events_gdf = build_event_geometries(events_df)
    empty_facilities = facilities.iloc[0:0]
    pairs2 = find_candidate_pairs(events_gdf, empty_facilities, association_radius_km=5.0)
    assert len(pairs2) == 0


def test_find_candidate_pairs_distance_never_negative() -> None:
    facilities = _facilities_gdf(
        [
            {"facility_id": "F1", "geometry_type": "Point", "geometry": Point(10.0, 10.0)},
            {"facility_id": "F2", "geometry_type": "Polygon", "geometry": Polygon([(10.0, 10.0), (10.02, 10.0), (10.02, 10.02), (10.0, 10.02)])},
        ]
    )
    events_df = pd.DataFrame(
        [
            _event_row("EVT_1", [(10.0, 10.0), (10.0001, 10.0001)]),
            _event_row("EVT_2", [(10.01, 10.01), (10.011, 10.011)]),
        ]
    )
    events_gdf = build_event_geometries(events_df)
    pairs = find_candidate_pairs(events_gdf, facilities, association_radius_km=5.0)
    assert (pairs["distance_km"] >= 0).all()


def test_india_equal_area_crs_is_a_valid_proj_string() -> None:
    # Sanity check that the CRS constant is actually usable by geopandas/pyproj.
    gdf = gpd.GeoDataFrame({"geometry": [Point(80.0, 20.0)]}, crs="EPSG:4326")
    projected = gdf.to_crs(INDIA_EQUAL_AREA_CRS)
    assert projected.crs is not None
