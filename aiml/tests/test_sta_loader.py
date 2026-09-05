"""Tests for STA loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.sta_evidence.config import LAYER_MASK, STAConfig
from src.sta_evidence.sta_loader import (
    STASourceMissingError,
    load_sta_vector,
    require_sta_sources,
    resolve_existing_paths,
)
from tests.fixtures.sta.make_fixtures import write_synthetic_sta_mask_geojson


def test_missing_source_raises(tmp_path: Path) -> None:
    config = STAConfig(
        mask_path=tmp_path / "missing_mask.geojson",
        detection_path=tmp_path / "missing_det.geojson",
    )
    with pytest.raises(STASourceMissingError):
        require_sta_sources(config)


def test_resolve_existing_paths(tmp_path: Path) -> None:
    mask = write_synthetic_sta_mask_geojson(tmp_path / "mask.geojson")
    config = STAConfig(mask_path=mask, detection_path=tmp_path / "nope.geojson")
    pairs = resolve_existing_paths(config)
    assert len(pairs) == 1
    assert pairs[0][1] == LAYER_MASK


def test_load_geojson_mask(tmp_path: Path) -> None:
    mask = write_synthetic_sta_mask_geojson(tmp_path / "mask.geojson")
    gdf = load_sta_vector(mask, LAYER_MASK)
    assert len(gdf) == 2
    assert str(gdf.crs).upper() in ("EPSG:4326", "EPSG:4326")
    assert (gdf["_sta_layer_type"] == LAYER_MASK).all()


def test_unsupported_format_raises(tmp_path: Path) -> None:
    bad = tmp_path / "sta.txt"
    bad.write_text("not a spatial file", encoding="utf-8")
    with pytest.raises(ValueError):
        load_sta_vector(bad, LAYER_MASK)


def test_load_csv_latlon(tmp_path: Path) -> None:
    csv_path = tmp_path / "sta.csv"
    csv_path.write_text("id,latitude,longitude\nA,28.0,77.0\nB,20.0,78.0\n", encoding="utf-8")
    gdf = load_sta_vector(csv_path, LAYER_MASK)
    assert len(gdf) == 2
    assert gdf.geometry.iloc[0].geom_type == "Point"
