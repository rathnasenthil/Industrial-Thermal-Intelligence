"""Tests for src.infrastructure.osm_loader (static OSM extract loading)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.infrastructure.osm_loader import (
    OsmInputError,
    discover_default_osm_input,
    load_osm_csv,
    load_osm_extract,
    load_osm_geojson,
)


def _write_geojson(path: Path, features: list[dict]) -> None:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")


def test_load_osm_geojson_extracts_combined_id_and_tags(tmp_path: Path) -> None:
    path = tmp_path / "extract.geojson"
    _write_geojson(
        path,
        [
            {
                "id": "way/111",
                "type": "Feature",
                "properties": {"industrial": "refinery", "name": "Test Refinery", "operator": "ACME"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[80.0, 20.0], [80.01, 20.0], [80.01, 20.01], [80.0, 20.01], [80.0, 20.0]]],
                },
            }
        ],
    )
    gdf = load_osm_geojson(path)

    assert len(gdf) == 1
    row = gdf.iloc[0]
    assert row["osm_id"] == "111"
    assert row["osm_type"] == "way"
    assert row["name"] == "Test Refinery"
    assert row["raw_tags"]["industrial"] == "refinery"
    assert row["raw_tags"]["operator"] == "ACME"
    assert row["geometry"].geom_type == "Polygon"
    assert str(gdf.crs).upper() == "EPSG:4326"


def test_load_osm_geojson_falls_back_when_no_id_available(tmp_path: Path) -> None:
    path = tmp_path / "extract.geojson"
    _write_geojson(
        path,
        [
            {
                "type": "Feature",
                "properties": {"landuse": "industrial"},
                "geometry": {"type": "Point", "coordinates": [79.0, 14.0]},
            }
        ],
    )
    gdf = load_osm_geojson(path)

    assert gdf.iloc[0]["osm_id"] is None
    assert gdf.iloc[0]["osm_type"] is None


def test_load_osm_geojson_does_not_leak_nan_into_tags_for_heterogeneous_features(tmp_path: Path) -> None:
    """Regression test: geopandas.read_file fills missing properties across
    the union of all features' keys with float NaN. Without careful
    filtering, that NaN previously leaked into `raw_tags` as a literal
    "nan" string and corrupted tag-based classification (e.g. a shop
    with no `power` tag at all was incorrectly seen as `power="nan"`,
    which is truthy)."""
    path = tmp_path / "extract.geojson"
    _write_geojson(
        path,
        [
            {
                "id": "node/1",
                "type": "Feature",
                "properties": {"power": "plant", "plant:source": "coal"},
                "geometry": {"type": "Point", "coordinates": [77.0, 12.0]},
            },
            {
                "id": "node/2",
                "type": "Feature",
                "properties": {"shop": "bakery"},
                "geometry": {"type": "Point", "coordinates": [78.0, 13.0]},
            },
        ],
    )
    gdf = load_osm_geojson(path)

    bakery_tags = gdf.loc[gdf["osm_id"] == "2", "raw_tags"].iloc[0]
    assert bakery_tags == {"shop": "bakery"}
    assert "power" not in bakery_tags
    assert "plant:source" not in bakery_tags


def test_load_osm_geojson_reprojects_non_wgs84_crs_to_4326(tmp_path: Path) -> None:
    import geopandas as gpd
    from shapely.geometry import Point

    # Build a GeoDataFrame in Web Mercator (EPSG:3857) and write it out --
    # the loader must reproject it back to WGS84 (EPSG:4326) on load.
    gdf_3857 = gpd.GeoDataFrame(
        {"osm_id": ["1"], "osm_type": ["node"], "name": ["X"], "industrial": ["mine"]},
        geometry=[Point(8_575_000, 1_365_000)],  # roughly India, in Web Mercator meters
        crs="EPSG:3857",
    )
    path = tmp_path / "mercator.geojson"
    gdf_3857.to_file(path, driver="GeoJSON")

    gdf = load_osm_geojson(path)

    assert str(gdf.crs).upper() == "EPSG:4326"
    lon, lat = gdf.iloc[0]["geometry"].x, gdf.iloc[0]["geometry"].y
    assert -180.0 <= lon <= 180.0
    assert -90.0 <= lat <= 90.0


def test_load_osm_geojson_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_osm_geojson(tmp_path / "missing.geojson")


def test_load_osm_geojson_invalid_content_raises_osm_input_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.geojson"
    path.write_text("this is not valid geojson at all {{{", encoding="utf-8")
    with pytest.raises(OsmInputError):
        load_osm_geojson(path)


def test_load_osm_csv_with_geometry_wkt_column(tmp_path: Path) -> None:
    path = tmp_path / "extract.csv"
    pd.DataFrame(
        [
            {
                "osm_id": "501",
                "osm_type": "way",
                "name": "Coal Mine A",
                "geometry_wkt": "POLYGON ((81.0 22.0, 81.01 22.0, 81.01 22.01, 81.0 22.01, 81.0 22.0))",
                "tags": json.dumps({"industrial": "mine"}),
            }
        ]
    ).to_csv(path, index=False)

    gdf = load_osm_csv(path)

    assert len(gdf) == 1
    assert gdf.iloc[0]["geometry"].geom_type == "Polygon"
    assert gdf.iloc[0]["raw_tags"] == {"industrial": "mine"}


def test_load_osm_csv_with_lat_lon_columns(tmp_path: Path) -> None:
    path = tmp_path / "extract.csv"
    pd.DataFrame(
        [{"osm_id": "222", "osm_type": "node", "latitude": 12.0, "longitude": 77.0, "power": "plant"}]
    ).to_csv(path, index=False)

    gdf = load_osm_csv(path)

    assert gdf.iloc[0]["geometry"].geom_type == "Point"
    assert gdf.iloc[0]["geometry"].x == pytest.approx(77.0)
    assert gdf.iloc[0]["geometry"].y == pytest.approx(12.0)
    # No explicit 'tags' column -> arbitrary extra columns become tags.
    assert gdf.iloc[0]["raw_tags"] == {"power": "plant"}


def test_load_osm_csv_without_geometry_source_raises(tmp_path: Path) -> None:
    path = tmp_path / "extract.csv"
    pd.DataFrame([{"osm_id": "1", "osm_type": "node", "name": "No geometry"}]).to_csv(path, index=False)

    with pytest.raises(OsmInputError):
        load_osm_csv(path)


def test_load_osm_csv_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_osm_csv(tmp_path / "missing.csv")


def test_load_osm_extract_dispatches_by_extension(tmp_path: Path) -> None:
    geojson_path = tmp_path / "a.geojson"
    _write_geojson(geojson_path, [{"id": "node/1", "type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [1.0, 2.0]}}])
    csv_path = tmp_path / "b.csv"
    pd.DataFrame([{"osm_id": "1", "osm_type": "node", "latitude": 1.0, "longitude": 2.0}]).to_csv(csv_path, index=False)

    assert len(load_osm_extract(geojson_path)) == 1
    assert len(load_osm_extract(csv_path)) == 1


def test_load_osm_extract_unsupported_extension_raises(tmp_path: Path) -> None:
    path = tmp_path / "extract.txt"
    path.write_text("irrelevant", encoding="utf-8")
    with pytest.raises(OsmInputError):
        load_osm_extract(path)


def test_discover_default_osm_input_returns_none_when_absent(tmp_path: Path) -> None:
    assert discover_default_osm_input(tmp_path) is None


def test_discover_default_osm_input_returns_none_for_missing_directory(tmp_path: Path) -> None:
    assert discover_default_osm_input(tmp_path / "does_not_exist") is None


def test_discover_default_osm_input_finds_matching_file(tmp_path: Path) -> None:
    (tmp_path / "unrelated.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    target = tmp_path / "osm_facilities_india.geojson"
    _write_geojson(target, [])

    found = discover_default_osm_input(tmp_path)

    assert found == target


def test_discover_default_osm_input_never_fabricates_or_downloads(tmp_path: Path) -> None:
    """Explicitly asserts the 'no production input' contract: an empty raw
    directory yields None, never a fabricated/synthetic file."""
    assert discover_default_osm_input(tmp_path) is None
    assert list(tmp_path.iterdir()) == []
