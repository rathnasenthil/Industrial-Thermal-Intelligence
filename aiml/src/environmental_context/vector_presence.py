"""
Generic vector presence / coverage / distance context for Stage I.6.

Used by vegetation, built-up, water, and agriculture modules.
Uses spatial-index joins — never an events × features dense matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd

from src.environmental_context.config import EnvironmentalContextConfig
from src.environmental_context.context_schema import unavailable_binary_context_frame
from src.environmental_context.geometry_utils import build_event_geodataframe, project_events
from src.environmental_context.vector_loader import open_vector
from src.infrastructure.association_geometry import INDIA_EQUAL_AREA_CRS


def compute_vector_presence_context(
    events_df: pd.DataFrame,
    config: EnvironmentalContextConfig,
    *,
    path,
    prefix: str,
) -> tuple[pd.DataFrame, dict]:
    """Compute presence/coverage/distance evidence for one vector layer."""
    event_ids = events_df["event_id"].astype(str)
    gdf = open_vector(path)
    meta = {"available": False, "path": str(path) if path else None, "feature_count": 0, "prefix": prefix}
    if gdf is None:
        return unavailable_binary_context_frame(event_ids, prefix), meta
    if gdf.empty:
        meta.update({"available": True, "feature_count": 0, "note": "layer_present_but_empty"})
        # Layer exists but empty → available=true but all present=false / distances null? 
        # Spec: missing evidence ≠ zero. Empty layer means we *know* there are no features
        # in the file — present=False and coverage=0 are legitimate observations of emptiness,
        # but distance remains null (no nearest feature).
        n = len(event_ids)
        return pd.DataFrame(
            {
                "event_id": event_ids.to_numpy(),
                f"{prefix}_context_available": np.full(n, True),
                f"{prefix}_present": np.full(n, False),
                f"{prefix}_coverage_fraction": np.zeros(n),
                f"distance_to_{prefix}_km": np.full(n, np.nan),
            }
        ), meta

    meta.update({"available": True, "feature_count": int(len(gdf))})
    result = _vector_context(events_df, gdf, config, prefix=prefix)
    return result, meta


def _vector_context(
    events_df: pd.DataFrame,
    features: gpd.GeoDataFrame,
    config: EnvironmentalContextConfig,
    *,
    prefix: str,
) -> pd.DataFrame:
    events_gdf = build_event_geodataframe(events_df)
    events_proj = project_events(events_gdf).reset_index(drop=True)
    feats_proj = features.to_crs(INDIA_EQUAL_AREA_CRS).reset_index(drop=True)

    local_m = float(config.context_buffer_km) * 1000.0
    broad_m = float(config.broad_context_buffer_km) * 1000.0

    # Buffered footprint for local coverage / presence.
    search = events_proj[["event_id", "geometry"]].copy()
    search["geometry"] = events_proj.geometry.buffer(local_m)

    joined = gpd.sjoin(search, feats_proj[["geometry"]], how="left", predicate="intersects")
    # Presence: any match with non-null index_right
    present_ids = set(joined.loc[joined["index_right"].notna(), "event_id"].astype(str))

    # Coverage fraction: intersection area / buffer area for events with hits (vectorized batch).
    coverage = pd.Series(np.nan, index=events_proj["event_id"].astype(str))
    if present_ids:
        hit_mask = events_proj["event_id"].astype(str).isin(present_ids)
        hit_events = events_proj.loc[hit_mask].copy()
        buffers = hit_events.geometry.buffer(local_m)
        # Union of intersecting features per event via overlay would be heavy;
        # approximate coverage as clipped intersection with dissolved features in buffer.
        # Efficient approach: for each hit event, area(buffer ∩ unary_union(feats in sjoin hits)).
        hits = joined.loc[joined["index_right"].notna(), ["event_id", "index_right"]]
        for eid, group in hits.groupby("event_id", sort=True):
            idxs = group["index_right"].astype(int).unique()
            geom_union = feats_proj.geometry.iloc[idxs].union_all()
            buf = buffers.loc[hit_events["event_id"].astype(str) == str(eid)]
            if buf.empty:
                continue
            b = buf.iloc[0]
            inter = b.intersection(geom_union)
            frac = float(inter.area / b.area) if b.area > 0 else np.nan
            coverage.loc[str(eid)] = max(0.0, min(1.0, frac))

    # Distance to nearest feature (centroid → features), capped by broad buffer via sjoin.
    broad_search = gpd.GeoDataFrame(
        {"event_id": events_proj["event_id"].to_numpy()},
        geometry=gpd.GeoSeries(events_proj["centroid_proj"].tolist(), crs=INDIA_EQUAL_AREA_CRS).buffer(broad_m),
        crs=INDIA_EQUAL_AREA_CRS,
    )
    broad_joined = gpd.sjoin(broad_search, feats_proj[["geometry"]], how="left", predicate="intersects")

    distances = pd.Series(np.nan, index=events_proj["event_id"].astype(str))
    if broad_joined["index_right"].notna().any():
        bj = broad_joined.loc[broad_joined["index_right"].notna()].copy()
        epos = events_proj.reset_index().set_index("event_id")
        for eid, group in bj.groupby("event_id", sort=True):
            cent = epos.loc[str(eid), "centroid_proj"] if str(eid) in epos.index else None
            if cent is None:
                continue
            if isinstance(cent, pd.Series):
                cent = cent.iloc[0]
            idxs = group["index_right"].astype(int).unique()
            dists = feats_proj.geometry.iloc[idxs].distance(cent)
            distances.loc[str(eid)] = float(dists.min()) / 1000.0

    n = len(events_proj)
    eids = events_proj["event_id"].astype(str).to_numpy()
    present = np.array([eid in present_ids for eid in eids], dtype=object)
    # present is boolean when available
    present_bool = np.array([eid in present_ids for eid in eids], dtype=bool)

    cov = coverage.reindex(eids).to_numpy()
    # For present events without computed coverage (shouldn't happen), leave nan not 0.
    # For absent events: coverage_fraction = 0 is a true observation of no intersection.
    cov = np.where(present_bool, cov, 0.0)

    dist = distances.reindex(eids).to_numpy()

    return pd.DataFrame(
        {
            "event_id": eids,
            f"{prefix}_context_available": np.full(n, True),
            f"{prefix}_present": present_bool,
            f"{prefix}_coverage_fraction": cov,
            f"distance_to_{prefix}_km": dist,
        }
    )
