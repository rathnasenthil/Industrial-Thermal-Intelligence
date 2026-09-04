"""
Noise/outlier annotation for GIFT Stage G.

ST-DBSCAN labels a detection as noise when it does not have enough
spatio-temporal neighbors (per `STDBSCANConfig.min_samples`) to be part of
a cluster, AND is not within `spatial_eps_km`/`temporal_eps_hours` of any
detection that does qualify. This is NOT evidence the detection is
spurious — it may simply be a single-pass, currently-uncorroborated
thermal detection (e.g. a short agricultural burn, or a real source that
happened to be detected only once due to cloud cover on adjacent
overpasses). Per the project's requirements, noise detections are never
deleted; they are preserved with an explanation of why they were not
clustered.
"""

from __future__ import annotations

import pandas as pd

from src.event_formation.config import STDBSCANConfig


def annotate_noise(noise_detections: pd.DataFrame, neighbor_counts: pd.Series, config: STDBSCANConfig) -> pd.DataFrame:
    """Attach neighbor-count and a human-readable reason to noise detections.

    Args:
        noise_detections: The subset of detections labeled as noise
            (cluster label == -1).
        neighbor_counts: Spatio-temporal neighbor counts (including self),
            aligned by index with the *original* (pre-split) detections
            DataFrame that `noise_detections` was sliced from.
        config: The clustering configuration used, so the reason text can
            cite the actual thresholds applied.

    Returns:
        A copy of ``noise_detections`` with two added columns:
        `spatiotemporal_neighbor_count` and `noise_reason`.
    """
    out = noise_detections.copy()
    counts = neighbor_counts.loc[out.index]
    out["spatiotemporal_neighbor_count"] = counts.astype(int)

    def _reason(count: int) -> str:
        if count <= 1:
            return (
                f"Spatially/temporally isolated: no other detection found within "
                f"{config.spatial_eps_km} km and {config.temporal_eps_hours} h."
            )
        return (
            f"Had {count} detection(s) (including itself) within "
            f"{config.spatial_eps_km} km / {config.temporal_eps_hours} h, below "
            f"min_samples={config.min_samples}, and no neighboring detection "
            f"qualified as a cluster core either."
        )

    out["noise_reason"] = out["spatiotemporal_neighbor_count"].apply(_reason)
    return out
