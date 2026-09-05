"""
Spatial/temporal event ↔ STA matching for GIFT Stage I.5.

Uses GeoPandas spatial-index joins (STRtree-backed) — never an
events × STA dense matrix. Distance/buffer/intersection work happens in
the same India-centered Albers Equal-Area CRS as Stage I.2
(``INDIA_EQUAL_AREA_CRS``), imported without modifying I.2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt as shapely_wkt
from shapely.geometry import Point

from src.infrastructure.association_geometry import INDIA_EQUAL_AREA_CRS
from src.sta_evidence.config import (
    LAYER_MASK,
    NO_STA_ASSOCIATION,
    STA_INTERSECTS_EVENT,
    STA_NEAR_EVENT,
    STAConfig,
    TEMPORAL_NEAR_EVENT_TIME,
    TEMPORAL_NOT_APPLICABLE,
    TEMPORAL_OUTSIDE_EVENT_TIME,
    TEMPORAL_SAME_PERIOD,
    TEMPORAL_UNKNOWN,
)

CANDIDATE_COLUMNS: tuple[str, ...] = (
    "event_id",
    "sta_id",
    "sta_layer_type",
    "relationship",
    "distance_km",
    "intersection_area_m2",
    "sta_temporal_relation",
)


def build_event_geometries(events_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Build event GeoDataFrame from Stage G footprint/centroid columns.

    Prefers ``footprint_wkt`` (detection envelope). Falls back to centroid.
    Does not invent geometry.
    """
    if "event_id" not in events_df.columns:
        raise ValueError("Events table missing event_id.")

    n = len(events_df)
    footprints: list = [None] * n
    centroids: list = [None] * n

    if "footprint_wkt" in events_df.columns:
        for i, value in enumerate(events_df["footprint_wkt"].tolist()):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            text = str(value).strip()
            if not text or text.lower() == "nan":
                continue
            try:
                footprints[i] = shapely_wkt.loads(text)
            except Exception:
                footprints[i] = None

    if "centroid_wkt" in events_df.columns:
        for i, value in enumerate(events_df["centroid_wkt"].tolist()):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            text = str(value).strip()
            if not text or text.lower() == "nan":
                continue
            try:
                centroids[i] = shapely_wkt.loads(text)
            except Exception:
                centroids[i] = None

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
            "event_start": events_df["event_start"].to_numpy() if "event_start" in events_df.columns else [None] * n,
            "event_end": events_df["event_end"].to_numpy() if "event_end" in events_df.columns else [None] * n,
            "centroid_geom": centroids,
        },
        geometry=footprints,
        crs="EPSG:4326",
    )


def find_sta_candidate_pairs(
    events_gdf: gpd.GeoDataFrame,
    sta_gdf: gpd.GeoDataFrame,
    config: STAConfig,
) -> pd.DataFrame:
    """Return candidate (event, STA) pairs with relationship and metrics."""
    if events_gdf.empty or sta_gdf.empty:
        return pd.DataFrame(columns=list(CANDIDATE_COLUMNS))

    events_valid = events_gdf[events_gdf.geometry.notna() & ~events_gdf.geometry.is_empty].copy()
    sta_valid = sta_gdf[sta_gdf.geometry.notna() & ~sta_gdf.geometry.is_empty].copy()
    if events_valid.empty or sta_valid.empty:
        return pd.DataFrame(columns=list(CANDIDATE_COLUMNS))

    # Ensure required STA columns exist.
    for col in ("sta_id", "sta_layer_type"):
        if col not in sta_valid.columns:
            raise ValueError(f"STA GeoDataFrame missing required column '{col}'.")
    if "observation_datetime" not in sta_valid.columns:
        sta_valid = sta_valid.copy()
        sta_valid["observation_datetime"] = None

    radius_m = float(config.association_radius_km) * 1000.0

    events_proj = events_valid.to_crs(INDIA_EQUAL_AREA_CRS).reset_index(drop=True)
    cent_series = gpd.GeoSeries(events_valid["centroid_geom"].tolist(), crs="EPSG:4326")
    # Fill missing centroids with footprint centroids.
    missing_cent = cent_series.isna() | gpd.GeoSeries(cent_series).is_empty
    if missing_cent.any():
        filled = events_valid.geometry.centroid
        cent_series = cent_series.copy()
        cent_series.loc[missing_cent.to_numpy()] = filled.loc[missing_cent.to_numpy()].to_numpy()
    events_proj["centroid_proj"] = cent_series.to_crs(INDIA_EQUAL_AREA_CRS).to_numpy()

    sta_proj = sta_valid.to_crs(INDIA_EQUAL_AREA_CRS).reset_index(drop=True)

    search = events_proj[["event_id", "geometry"]].copy()
    search["geometry"] = events_proj.geometry.buffer(radius_m)

    joined = gpd.sjoin(
        search,
        sta_proj[["sta_id", "sta_layer_type", "observation_datetime", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    if joined.empty:
        return pd.DataFrame(columns=list(CANDIDATE_COLUMNS))

    # Left index = event position; index_right = STA position (both reset 0..n-1).
    event_pos = joined.index.to_numpy()
    sta_pos = joined["index_right"].to_numpy()

    event_footprints = gpd.GeoSeries(events_proj.geometry.to_numpy()[event_pos], crs=INDIA_EQUAL_AREA_CRS)
    event_centroids = gpd.GeoSeries(events_proj["centroid_proj"].to_numpy()[event_pos], crs=INDIA_EQUAL_AREA_CRS)
    sta_geoms = gpd.GeoSeries(sta_proj.geometry.to_numpy()[sta_pos], crs=INDIA_EQUAL_AREA_CRS)

    intersects = event_footprints.intersects(sta_geoms)
    distances_m = event_centroids.distance(sta_geoms)
    intersection_area = event_footprints.intersection(sta_geoms).area

    relationships = np.where(
        intersects.to_numpy(),
        STA_INTERSECTS_EVENT,
        np.where(distances_m.to_numpy() <= radius_m, STA_NEAR_EVENT, NO_STA_ASSOCIATION),
    )
    keep = relationships != NO_STA_ASSOCIATION

    event_starts = events_proj["event_start"].to_numpy()[event_pos]
    event_ends = events_proj["event_end"].to_numpy()[event_pos]
    sta_obs = sta_proj["observation_datetime"].to_numpy()[sta_pos]
    layer_types = sta_proj["sta_layer_type"].to_numpy()[sta_pos]

    temporal = np.array(
        [
            classify_temporal_relation(
                layer_type=str(layer_types[i]),
                observation_datetime=sta_obs[i],
                event_start=event_starts[i],
                event_end=event_ends[i],
                near_hours=config.near_event_time_hours,
            )
            for i in range(len(relationships))
        ],
        dtype=object,
    )

    out = pd.DataFrame(
        {
            "event_id": joined["event_id"].to_numpy(),
            "sta_id": joined["sta_id"].to_numpy(),
            "sta_layer_type": layer_types,
            "relationship": relationships,
            "distance_km": distances_m.to_numpy() / 1000.0,
            "intersection_area_m2": intersection_area.to_numpy(),
            "sta_temporal_relation": temporal,
        }
    )
    out = out.loc[keep].copy()
    out.loc[out["relationship"] != STA_INTERSECTS_EVENT, "intersection_area_m2"] = np.nan
    return out.reset_index(drop=True)


def classify_temporal_relation(
    *,
    layer_type: str,
    observation_datetime: object,
    event_start: object,
    event_end: object,
    near_hours: float,
) -> str:
    """Temporal relation for one STA feature vs one event."""
    if str(layer_type).upper() == LAYER_MASK:
        return TEMPORAL_NOT_APPLICABLE
    if observation_datetime is None or (isinstance(observation_datetime, float) and np.isnan(observation_datetime)):
        return TEMPORAL_UNKNOWN
    text = str(observation_datetime).strip()
    if text == "" or text.lower() == "nan":
        return TEMPORAL_UNKNOWN
    try:
        obs = pd.to_datetime(observation_datetime, utc=True)
        start = pd.to_datetime(event_start, utc=True)
        end = pd.to_datetime(event_end, utc=True)
    except (TypeError, ValueError):
        return TEMPORAL_UNKNOWN
    if pd.isna(obs) or pd.isna(start) or pd.isna(end):
        return TEMPORAL_UNKNOWN
    if start <= obs <= end:
        return TEMPORAL_SAME_PERIOD
    delta_hours = min(abs((obs - start).total_seconds()), abs((obs - end).total_seconds())) / 3600.0
    if delta_hours <= near_hours:
        return TEMPORAL_NEAR_EVENT_TIME
    return TEMPORAL_OUTSIDE_EVENT_TIME
