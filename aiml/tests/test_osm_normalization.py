"""Tests for src.infrastructure.osm_normalization (tag normalization + canonical frame)."""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon

from src.infrastructure.config import InfrastructureConfig
from src.infrastructure.facility_schema import (
    INDUSTRIAL_AREA,
    LNG_TERMINAL,
    MINE,
    OTHER_INDUSTRIAL,
    POWER_PLANT,
    REFINERY,
    UNKNOWN,
)
from src.infrastructure.osm_normalization import classify_facility_type, normalize_osm_facilities

# ---------------------------------------------------------------------------
# classify_facility_type
# ---------------------------------------------------------------------------


def test_refinery_mapping_from_industrial_tag() -> None:
    result = classify_facility_type({"industrial": "refinery"})
    assert result.facility_type == REFINERY
    assert result.confidence == "high"


def test_power_plant_mapping_from_power_plant_tag() -> None:
    result = classify_facility_type({"power": "plant", "plant:source": "coal"})
    assert result.facility_type == POWER_PLANT
    assert result.power_type == "coal"
    assert result.confidence == "high"


def test_power_substation_is_not_a_power_plant() -> None:
    """power=substation/line/tower are grid infrastructure, not generation
    facilities -- must not be overgeneralized into POWER_PLANT."""
    result = classify_facility_type({"power": "substation"})
    assert result.facility_type != POWER_PLANT


def test_mine_mapping_from_industrial_mine_tag() -> None:
    result = classify_facility_type({"industrial": "mine"})
    assert result.facility_type == MINE
    assert result.confidence == "high"


def test_mine_mapping_from_landuse_quarry_tag() -> None:
    result = classify_facility_type({"landuse": "quarry"})
    assert result.facility_type == MINE
    assert result.industrial_subtype == "quarry"


def test_industrial_area_mapping_from_landuse_industrial() -> None:
    result = classify_facility_type({"landuse": "industrial"})
    assert result.facility_type == INDUSTRIAL_AREA
    assert result.confidence == "high"


def test_lng_terminal_requires_both_gas_evidence_and_name_match() -> None:
    result = classify_facility_type({"industrial": "gas"}, name="Dahej LNG Terminal")
    assert result.facility_type == LNG_TERMINAL
    assert result.confidence == "medium"


def test_generic_gas_facility_without_lng_name_is_not_lng_terminal() -> None:
    """OSM has no single universal LNG tag; without explicit name
    corroboration this must NOT be overgeneralized to LNG_TERMINAL."""
    result = classify_facility_type({"industrial": "gas"}, name="Some Gas Plant")
    assert result.facility_type != LNG_TERMINAL
    assert result.facility_type == OTHER_INDUSTRIAL


def test_unmapped_industrial_tag_is_other_industrial() -> None:
    result = classify_facility_type({"industrial": "factory"})
    assert result.facility_type == OTHER_INDUSTRIAL
    assert result.confidence == "medium"


def test_unrelated_tags_are_unknown() -> None:
    result = classify_facility_type({"shop": "bakery", "amenity": "cafe"})
    assert result.facility_type == UNKNOWN
    assert result.confidence is None


def test_empty_tags_are_unknown() -> None:
    result = classify_facility_type({})
    assert result.facility_type == UNKNOWN


def test_refinery_takes_priority_over_generic_landuse_industrial() -> None:
    """An object tagged both landuse=industrial and industrial=refinery
    must resolve to the more specific REFINERY, not INDUSTRIAL_AREA."""
    result = classify_facility_type({"landuse": "industrial", "industrial": "refinery"})
    assert result.facility_type == REFINERY


# ---------------------------------------------------------------------------
# normalize_osm_facilities
# ---------------------------------------------------------------------------

_CONFIG = InfrastructureConfig()


def _raw_gdf(rows: list[dict]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def test_point_geometry_is_preserved() -> None:
    raw = _raw_gdf(
        [{"osm_id": "1", "osm_type": "node", "name": "P", "raw_tags": {"power": "plant"}, "geometry": Point(77.0, 12.0)}]
    )
    result = normalize_osm_facilities(raw, _CONFIG, "v1")
    assert result.iloc[0]["geometry_type"] == "Point"
    assert result.iloc[0]["latitude"] == pytest.approx(12.0)
    assert result.iloc[0]["longitude"] == pytest.approx(77.0)


def test_polygon_geometry_is_preserved_not_replaced_by_centroid() -> None:
    polygon = Polygon([(80.0, 20.0), (80.02, 20.0), (80.02, 20.02), (80.0, 20.02)])
    raw = _raw_gdf(
        [{"osm_id": "2", "osm_type": "way", "name": "Poly", "raw_tags": {"landuse": "industrial"}, "geometry": polygon}]
    )
    result = normalize_osm_facilities(raw, _CONFIG, "v1")
    row = result.iloc[0]
    assert row["geometry_type"] == "Polygon"
    # Original polygon geometry is preserved (not collapsed to a point).
    assert result.geometry.iloc[0].equals(polygon)
    # Representative point is the centroid, used only for spatial association.
    assert row["latitude"] == pytest.approx(polygon.centroid.y)
    assert row["longitude"] == pytest.approx(polygon.centroid.x)


def test_multipolygon_geometry_is_preserved() -> None:
    poly_a = Polygon([(81.0, 22.0), (81.01, 22.0), (81.01, 22.01), (81.0, 22.01)])
    poly_b = Polygon([(81.5, 22.5), (81.51, 22.5), (81.51, 22.51), (81.5, 22.51)])
    multi = MultiPolygon([poly_a, poly_b])
    raw = _raw_gdf(
        [{"osm_id": "3", "osm_type": "relation", "name": "Multi", "raw_tags": {"industrial": "mine"}, "geometry": multi}]
    )
    result = normalize_osm_facilities(raw, _CONFIG, "v1")
    assert result.iloc[0]["geometry_type"] == "MultiPolygon"
    assert result.iloc[0]["facility_type"] == MINE


def test_deterministic_facility_id_from_osm_identifier() -> None:
    raw = _raw_gdf(
        [{"osm_id": "123456789", "osm_type": "way", "name": None, "raw_tags": {}, "geometry": Point(1.0, 2.0)}]
    )
    result = normalize_osm_facilities(raw, _CONFIG, "v1")
    assert result.iloc[0]["facility_id"] == "osm_way_123456789"


def test_deterministic_fallback_facility_id_when_osm_identifier_missing() -> None:
    raw = _raw_gdf(
        [{"osm_id": None, "osm_type": None, "name": "No ID", "raw_tags": {"landuse": "industrial"}, "geometry": Point(3.0, 4.0)}]
    )
    result_1 = normalize_osm_facilities(raw, _CONFIG, "v1")
    result_2 = normalize_osm_facilities(raw, _CONFIG, "v1")

    facility_id = result_1.iloc[0]["facility_id"]
    assert facility_id.startswith("fallback_")
    assert facility_id == result_2.iloc[0]["facility_id"]  # deterministic


def test_missing_optional_attributes_remain_null_not_fabricated() -> None:
    raw = _raw_gdf(
        [{"osm_id": "1", "osm_type": "node", "name": None, "raw_tags": {"power": "plant"}, "geometry": Point(1.0, 1.0)}]
    )
    result = normalize_osm_facilities(raw, _CONFIG, "v1")
    row = result.iloc[0]
    assert row["facility_name"] is None
    assert row["operator"] is None  # no 'operator' tag given -> not invented.


def test_original_osm_tags_are_preserved_verbatim_in_osm_tags_column() -> None:
    raw = _raw_gdf(
        [
            {
                "osm_id": "1",
                "osm_type": "node",
                "name": "X",
                "raw_tags": {"power": "plant", "plant:source": "gas", "operator": "NTPC"},
                "geometry": Point(1.0, 1.0),
            }
        ]
    )
    result = normalize_osm_facilities(raw, _CONFIG, "v1")
    import json

    tags = json.loads(result.iloc[0]["osm_tags"])
    assert tags == {"power": "plant", "plant:source": "gas", "operator": "NTPC"}
    assert result.iloc[0]["operator"] == "NTPC"


def test_source_and_source_version_are_recorded() -> None:
    raw = _raw_gdf([{"osm_id": "1", "osm_type": "node", "name": None, "raw_tags": {}, "geometry": Point(1.0, 1.0)}])
    result = normalize_osm_facilities(raw, _CONFIG, "my_extract_v2")
    assert result.iloc[0]["source"] == _CONFIG.source_label
    assert result.iloc[0]["source_version"] == "my_extract_v2"


def test_normalize_never_drops_or_reorders_rows() -> None:
    raw = _raw_gdf(
        [
            {"osm_id": "1", "osm_type": "node", "name": "A", "raw_tags": {}, "geometry": Point(1.0, 1.0)},
            {"osm_id": "2", "osm_type": "node", "name": "B", "raw_tags": {}, "geometry": None},
            {"osm_id": "3", "osm_type": "node", "name": "C", "raw_tags": {}, "geometry": Point(3.0, 3.0)},
        ]
    )
    result = normalize_osm_facilities(raw, _CONFIG, "v1")
    assert len(result) == 3
    assert list(result["osm_id"]) == ["1", "2", "3"]
