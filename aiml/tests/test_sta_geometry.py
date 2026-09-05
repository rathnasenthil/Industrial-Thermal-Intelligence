"""CRS and projected-distance tests for Stage I.5."""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import Point

from src.infrastructure.association_geometry import INDIA_EQUAL_AREA_CRS
from src.sta_evidence.config import STAConfig
from src.sta_evidence.sta_matching import build_event_geometries, find_sta_candidate_pairs
from src.sta_evidence.sta_normalization import canonical_to_geodataframe, normalize_sta_geodataframe


def test_india_crs_used_for_distance_not_degrees() -> None:
    # Two points ~1 km apart in projected meters near Delhi-ish coords.
    sta = gpd.GeoDataFrame(
        {"id": ["S1"], "_sta_layer_type": ["MASK"], "observation_datetime": [None]},
        geometry=[Point(77.0, 28.0)],
        crs="EPSG:4326",
    )
    canonical, _ = normalize_sta_geodataframe(sta, STAConfig())
    sta_gdf = canonical_to_geodataframe(canonical)

    import pandas as pd

    events = pd.DataFrame(
        [
            {
                "event_id": "E1",
                "event_start": "2023-01-01T00:00:00+00:00",
                "event_end": "2023-01-01T01:00:00+00:00",
                "centroid_latitude": 28.005,
                "centroid_longitude": 77.0,
                "centroid_wkt": "POINT (77.0 28.005)",
                "footprint_wkt": "POINT (77.0 28.005)",
            }
        ]
    )
    pairs = find_sta_candidate_pairs(build_event_geometries(events), sta_gdf, STAConfig(association_radius_km=1.0))
    assert not pairs.empty
    # Degree difference is 0.005; if wrongly treated as km that would be tiny wrong scale.
    # Projected distance should be on the order of ~0.5 km.
    assert 0.2 < float(pairs.iloc[0]["distance_km"]) < 0.9


def test_storage_crs_is_4326() -> None:
    sta = gpd.GeoDataFrame(
        {"id": ["S1"], "_sta_layer_type": ["MASK"]},
        geometry=[Point(77.0, 28.0)],
        crs="EPSG:4326",
    )
    canonical, _ = normalize_sta_geodataframe(sta, STAConfig())
    gdf = canonical_to_geodataframe(canonical)
    assert gdf.crs.to_epsg() == 4326
    assert "aea" in INDIA_EQUAL_AREA_CRS
