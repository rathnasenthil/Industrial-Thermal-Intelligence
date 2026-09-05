"""Synthetic fixtures for Stage I.6 tests (NOT production environmental data)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon


def make_synthetic_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "EVT_A",
                "centroid_latitude": 28.01,
                "centroid_longitude": 77.01,
                "centroid_wkt": "POINT (77.01 28.01)",
                "footprint_wkt": "POINT (77.01 28.01)",
                "event_start": "2023-01-01T00:00:00+00:00",
                "event_end": "2023-01-01T01:00:00+00:00",
                "anomaly_score": 1.2,
                "anomaly_status": "NORMAL",
                "anomaly_confidence": "MEDIUM",
                "peak_frp_deviation": 0.5,
                "event_size_deviation": 0.1,
                "duration_deviation": 0.2,
                "distance_deviation": None,
                "persistence_deviation": 0.0,
                "monthly_deviation": None,
                "sta_association_status": "NO_STA_ASSOCIATION",
                "sta_evidence_quality": "NONE",
            },
            {
                "event_id": "EVT_B",
                "centroid_latitude": 20.0,
                "centroid_longitude": 78.0,
                "centroid_wkt": "POINT (78.0 20.0)",
                "footprint_wkt": "POINT (78.0 20.0)",
                "event_start": "2023-02-01T00:00:00+00:00",
                "event_end": "2023-02-01T01:00:00+00:00",
                "anomaly_score": None,
                "anomaly_status": "INSUFFICIENT_HISTORY",
                "anomaly_confidence": "NONE",
                "peak_frp_deviation": None,
                "event_size_deviation": None,
                "duration_deviation": None,
                "distance_deviation": None,
                "persistence_deviation": None,
                "monthly_deviation": None,
                "sta_association_status": "STA_ASSOCIATED",
                "sta_evidence_quality": "HIGH",
            },
        ]
    )


def write_water_geojson(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(
        {"name": ["lake"]},
        geometry=[Polygon([(77.0, 28.0), (77.02, 28.0), (77.02, 28.02), (77.0, 28.02)])],
        crs="EPSG:4326",
    )
    gdf.to_file(path, driver="GeoJSON")
    return path


def write_landcover_raster(path: Path) -> Path:
    """Tiny EPSG:4326 categorical GeoTIFF for tests."""
    import rasterio
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    # 10x10 grid covering ~76.99-77.03 lon, 27.99-28.03 lat
    data = np.full((10, 10), 2, dtype=np.uint8)  # class 2
    data[4:7, 4:7] = 5  # class 5 near 77.01, 28.01
    transform = from_origin(76.99, 28.03, 0.004, 0.004)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(data, 1)
    return path
