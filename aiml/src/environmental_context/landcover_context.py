"""
Land-cover context evidence for Stage I.6.

Supports a local categorical raster (GeoTIFF) and/or a local vector layer.
Class IDs are never treated as continuous quantities. Missing data → unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd

from src.environmental_context.config import EnvironmentalContextConfig
from src.environmental_context.context_schema import unavailable_landcover_frame
from src.environmental_context.geometry_utils import build_event_geodataframe
from src.environmental_context.raster_loader import open_raster
from src.environmental_context.vector_loader import open_vector
from src.infrastructure.association_geometry import INDIA_EQUAL_AREA_CRS


def compute_landcover_context(events_df: pd.DataFrame, config: EnvironmentalContextConfig) -> tuple[pd.DataFrame, dict]:
    """Return per-event land-cover evidence + dataset status metadata."""
    event_ids = events_df["event_id"].astype(str)
    raster = open_raster(config.landcover_raster_path, source_name=config.landcover_source_name)
    vector = open_vector(config.landcover_vector_path)

    meta = {
        "source_type": None,
        "path": None,
        "available": False,
        "records_or_pixels": None,
    }

    if raster is not None:
        try:
            result = _from_raster(events_df, raster, config)
            meta.update(
                {
                    "source_type": "raster",
                    "path": str(raster.path),
                    "available": True,
                    "records_or_pixels": int(raster.width * raster.height),
                    "crs": str(raster.crs),
                }
            )
            return result, meta
        finally:
            raster.close()

    if vector is not None and not vector.empty:
        result = _from_vector(events_df, vector, config)
        meta.update(
            {
                "source_type": "vector",
                "path": str(config.landcover_vector_path),
                "available": True,
                "records_or_pixels": int(len(vector)),
            }
        )
        return result, meta

    return unavailable_landcover_frame(event_ids), meta


def _map_class(raw: object, class_map: dict[str, str]) -> str | None:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    key = str(int(raw)) if isinstance(raw, (int, np.integer, float)) and float(raw).is_integer() else str(raw)
    if key.lower() == "nan":
        return None
    return class_map.get(key, key)


def _from_raster(events_df: pd.DataFrame, raster, config: EnvironmentalContextConfig) -> pd.DataFrame:
    """Sample categorical land-cover at event centroids (EPSG:4326 → raster CRS)."""
    import pyproj
    from pyproj import Transformer

    lons = pd.to_numeric(events_df["centroid_longitude"], errors="coerce").to_numpy()
    lats = pd.to_numeric(events_df["centroid_latitude"], errors="coerce").to_numpy()

    # Transform lon/lat to raster CRS if needed.
    if raster.crs is not None and str(raster.crs).upper() not in ("EPSG:4326", "OGC:CRS84"):
        transformer = Transformer.from_crs("EPSG:4326", raster.crs, always_xy=True)
        xs, ys = transformer.transform(lons, lats)
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
    else:
        xs, ys = lons.astype(float), lats.astype(float)

    values = raster.sample_values(xs, ys)
    classes = [_map_class(v, config.landcover_class_map) if not np.isnan(v) else None for v in values]
    available = np.array([c is not None for c in classes], dtype=bool)

    return pd.DataFrame(
        {
            "event_id": events_df["event_id"].astype(str).to_numpy(),
            "landcover_available": available,
            "landcover_source": np.where(available, raster.source_name, None),
            "landcover_year": np.where(available, config.landcover_year, None),
            "dominant_landcover_class": classes,
            # Point sample → fraction 1.0 when available, else null (not 0).
            "dominant_landcover_fraction": np.where(available, 1.0, np.nan),
            "landcover_class_count": np.where(available, 1.0, np.nan),
        }
    )


def _from_vector(events_df: pd.DataFrame, landcover_gdf: gpd.GeoDataFrame, config: EnvironmentalContextConfig) -> pd.DataFrame:
    """Assign land-cover class via spatial join of centroid to polygons (indexed)."""
    events_gdf = build_event_geodataframe(events_df)
    # Prefer a class column if present.
    class_col = next(
        (c for c in ("landcover_class", "class", "class_name", "LC_class", "name") if c in landcover_gdf.columns),
        None,
    )
    lc = landcover_gdf.copy()
    if class_col is None:
        lc["landcover_class"] = "UNKNOWN"
        class_col = "landcover_class"

    cents = gpd.GeoDataFrame(
        {"event_id": events_gdf["event_id"]},
        geometry=gpd.GeoSeries(events_gdf["centroid_geom"].tolist(), crs="EPSG:4326"),
        crs="EPSG:4326",
    )
    cents = cents[cents.geometry.notna()].copy()
    joined = gpd.sjoin(cents, lc[[class_col, "geometry"]], how="left", predicate="intersects")
    # Deterministic: if multiple hits, keep first by sorted index of right
    joined = joined.sort_values(["event_id", "index_right"], kind="mergesort")
    joined = joined.drop_duplicates("event_id", keep="first")

    out = unavailable_landcover_frame(events_df["event_id"])
    out = out.set_index("event_id")
    hit = joined.set_index("event_id")
    for eid, row in hit.iterrows():
        raw = row.get(class_col)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            continue
        label = _map_class(raw, config.landcover_class_map)
        if label is None:
            continue
        out.at[eid, "landcover_available"] = True
        out.at[eid, "landcover_source"] = config.landcover_source_name
        out.at[eid, "landcover_year"] = config.landcover_year
        out.at[eid, "dominant_landcover_class"] = label
        out.at[eid, "dominant_landcover_fraction"] = 1.0
        out.at[eid, "landcover_class_count"] = 1.0
    return out.reset_index()
