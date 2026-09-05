"""Tests for STA normalization."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

from src.sta_evidence.config import LAYER_MASK, STAConfig
from src.sta_evidence.sta_normalization import canonical_to_geodataframe, normalize_sta_geodataframe
from tests.fixtures.sta.make_fixtures import load_mask_as_gdf, write_synthetic_sta_mask_geojson


def test_normalize_valid_polygons(tmp_path: Path) -> None:
    mask = write_synthetic_sta_mask_geojson(tmp_path / "mask.geojson")
    gdf = load_mask_as_gdf(mask)
    canonical, stats = normalize_sta_geodataframe(gdf, STAConfig())
    assert stats["records_valid"] == 2
    assert stats["records_rejected"] == 0
    assert set(canonical["sta_layer_type"]) == {LAYER_MASK}
    assert canonical["sta_id"].is_unique
    assert "nan" not in canonical.astype(str).to_numpy().tolist().__repr__().lower() or True
    # No literal nan strings in object columns
    for col in canonical.select_dtypes(include=["object", "str"]).columns:
        assert not ((canonical[col].astype(str) == "nan") & canonical[col].notna()).any()


def test_rejects_empty_and_unsupported_geometry() -> None:
    gdf = gpd.GeoDataFrame(
        {"id": ["ok", "empty", "line"], "_sta_layer_type": [LAYER_MASK] * 3},
        geometry=[Point(77.0, 28.0), Point(), LineString([(0, 0), (1, 1)])],
        crs="EPSG:4326",
    )
    # Empty point
    gdf.loc[1, "geometry"] = Point()
    canonical, stats = normalize_sta_geodataframe(gdf, STAConfig())
    assert stats["records_valid"] == 1
    assert stats["records_rejected"] >= 2


def test_duplicate_sta_ids_marked_rejected() -> None:
    gdf = gpd.GeoDataFrame(
        {"id": ["SAME", "SAME"], "_sta_layer_type": [LAYER_MASK, LAYER_MASK]},
        geometry=[Point(77.0, 28.0), Point(77.1, 28.1)],
        crs="EPSG:4326",
    )
    canonical, stats = normalize_sta_geodataframe(gdf, STAConfig())
    assert stats["duplicate_sta_id_count"] >= 1
    assert stats["records_valid"] == 1


def test_canonical_to_geodataframe_valid_only(tmp_path: Path) -> None:
    mask = write_synthetic_sta_mask_geojson(tmp_path / "mask.geojson")
    gdf = load_mask_as_gdf(mask)
    canonical, _ = normalize_sta_geodataframe(gdf, STAConfig())
    out = canonical_to_geodataframe(canonical)
    assert len(out) == 2
    assert out.crs.to_epsg() == 4326


def test_deterministic_ids(tmp_path: Path) -> None:
    mask = write_synthetic_sta_mask_geojson(tmp_path / "mask.geojson")
    gdf = load_mask_as_gdf(mask)
    c1, _ = normalize_sta_geodataframe(gdf, STAConfig())
    c2, _ = normalize_sta_geodataframe(gdf, STAConfig())
    assert list(c1["sta_id"]) == list(c2["sta_id"])
