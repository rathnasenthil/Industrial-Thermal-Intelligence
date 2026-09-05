"""
Shared event-geometry helpers for Stage I.6 context modules.

Uses Stage G footprint/centroid. Detection envelopes are not fire perimeters.
Projected operations reuse I.2's India-centered Albers CRS (imported only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt as shapely_wkt
from shapely.geometry import Point

from src.infrastructure.association_geometry import INDIA_EQUAL_AREA_CRS


def build_event_geodataframe(events_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Build EPSG:4326 GeoDataFrame with footprint geometry + centroid_geom."""
    n = len(events_df)
    footprints: list = [None] * n
    centroids: list = [None] * n

    if "footprint_wkt" in events_df.columns:
        for i, value in enumerate(events_df["footprint_wkt"].tolist()):
            footprints[i] = _parse_wkt(value)
    if "centroid_wkt" in events_df.columns:
        for i, value in enumerate(events_df["centroid_wkt"].tolist()):
            centroids[i] = _parse_wkt(value)

    if "centroid_latitude" in events_df.columns and "centroid_longitude" in events_df.columns:
        lats = pd.to_numeric(events_df["centroid_latitude"], errors="coerce").to_numpy()
        lons = pd.to_numeric(events_df["centroid_longitude"], errors="coerce").to_numpy()
        for i in range(n):
            if centroids[i] is None and not np.isnan(lats[i]) and not np.isnan(lons[i]):
                centroids[i] = Point(float(lons[i]), float(lats[i]))

    for i in range(n):
        if footprints[i] is None and centroids[i] is not None:
            footprints[i] = centroids[i]

    return gpd.GeoDataFrame(
        {
            "event_id": events_df["event_id"].astype(str).to_numpy(),
            "centroid_geom": centroids,
        },
        geometry=footprints,
        crs="EPSG:4326",
    )


def project_events(events_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject footprints and centroids to India Albers (meters)."""
    out = events_gdf.to_crs(INDIA_EQUAL_AREA_CRS).copy()
    cents = gpd.GeoSeries(events_gdf["centroid_geom"].tolist(), crs="EPSG:4326")
    missing = cents.isna()
    if missing.any():
        filled = events_gdf.geometry.centroid
        cents = cents.copy()
        cents.loc[missing.to_numpy()] = filled.loc[missing.to_numpy()].to_numpy()
    out["centroid_proj"] = cents.to_crs(INDIA_EQUAL_AREA_CRS).to_numpy()
    return out


def _parse_wkt(value: object):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return shapely_wkt.loads(text)
    except Exception:
        return None
