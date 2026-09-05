"""
Local vector loaders for Stage I.6.

Supports GeoJSON / GPKG / Shapefile / CSV (lat/lon or WKT). Returns None when
the configured path is missing. Never fabricates geometries.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import wkt as shapely_wkt

from src.environmental_context.raster_loader import resolve_existing_path

SUPPORTED_SUFFIXES = frozenset({".geojson", ".json", ".gpkg", ".shp", ".csv"})


def open_vector(path: Path | None) -> gpd.GeoDataFrame | None:
    """Load a local vector layer as EPSG:4326, or None if absent/unreadable."""
    existing = resolve_existing_path(path)
    if existing is None:
        return None
    suffix = existing.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        return None
    try:
        if suffix == ".csv":
            gdf = _load_csv(existing)
        else:
            gdf = gpd.read_file(existing)
    except Exception:
        return None
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def _load_csv(path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(path)
    if "geometry_wkt" in df.columns or "wkt" in df.columns:
        col = "geometry_wkt" if "geometry_wkt" in df.columns else "wkt"
        geoms = []
        for value in df[col]:
            if pd.isna(value) or str(value).strip() == "" or str(value).lower() == "nan":
                geoms.append(None)
            else:
                geoms.append(shapely_wkt.loads(str(value)))
        return gpd.GeoDataFrame(df.drop(columns=[col]), geometry=geoms, crs="EPSG:4326")
    lat = next((c for c in ("latitude", "lat") if c in df.columns), None)
    lon = next((c for c in ("longitude", "lon", "lng") if c in df.columns), None)
    if lat is None or lon is None:
        raise ValueError(f"CSV vector {path} needs geometry_wkt/wkt or latitude+longitude.")
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs="EPSG:4326")
