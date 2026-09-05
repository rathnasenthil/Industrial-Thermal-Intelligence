"""
Satellite-derived contextual evidence abstraction (Stage I.6).

No live API. Loads a local raster if configured and present; otherwise
reports satellite_context_available=false without fabricating values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.environmental_context.config import EnvironmentalContextConfig
from src.environmental_context.context_schema import unavailable_satellite_frame
from src.environmental_context.raster_loader import open_raster
from pyproj import Transformer


def compute_satellite_context(events_df: pd.DataFrame, config: EnvironmentalContextConfig) -> tuple[pd.DataFrame, dict]:
    event_ids = events_df["event_id"].astype(str)
    raster = open_raster(config.satellite_raster_path, source_name="local_satellite_context")
    meta = {"available": False, "path": str(config.satellite_raster_path) if config.satellite_raster_path else None}
    if raster is None:
        return unavailable_satellite_frame(event_ids), meta

    try:
        lons = pd.to_numeric(events_df["centroid_longitude"], errors="coerce").to_numpy()
        lats = pd.to_numeric(events_df["centroid_latitude"], errors="coerce").to_numpy()
        if raster.crs is not None and str(raster.crs).upper() not in ("EPSG:4326", "OGC:CRS84"):
            transformer = Transformer.from_crs("EPSG:4326", raster.crs, always_xy=True)
            xs, ys = transformer.transform(lons, lats)
            xs = np.asarray(xs, dtype=float)
            ys = np.asarray(ys, dtype=float)
        else:
            xs, ys = lons.astype(float), lats.astype(float)
        values = raster.sample_values(xs, ys)
        available = ~np.isnan(values)
        meta.update(
            {
                "available": True,
                "path": str(raster.path),
                "crs": str(raster.crs),
                "sampled_valid_count": int(available.sum()),
            }
        )
        return pd.DataFrame(
            {
                "event_id": event_ids.to_numpy(),
                "satellite_context_available": available,
                "satellite_source": np.where(available, raster.source_name, None),
                "satellite_value": values,
                "satellite_value_name": np.where(available, "band_1", None),
            }
        ), meta
    finally:
        raster.close()
