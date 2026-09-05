"""
Canonical I.6 environmental-context field schema and unavailable-row builders.

Missing evidence → null + availability=False.
Never emit fabricated zeros for unavailable numeric evidence.
Never emit literal \"nan\" strings.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

LANDCOVER_COLUMNS: tuple[str, ...] = (
    "landcover_available",
    "landcover_source",
    "landcover_year",
    "dominant_landcover_class",
    "dominant_landcover_fraction",
    "landcover_class_count",
)

VEGETATION_COLUMNS: tuple[str, ...] = (
    "vegetation_context_available",
    "vegetation_present",
    "vegetation_coverage_fraction",
    "distance_to_vegetation_km",
)

BUILTUP_COLUMNS: tuple[str, ...] = (
    "builtup_context_available",
    "builtup_present",
    "builtup_coverage_fraction",
    "distance_to_builtup_km",
)

WATER_COLUMNS: tuple[str, ...] = (
    "water_context_available",
    "water_present",
    "water_coverage_fraction",
    "distance_to_water_km",
)

AGRICULTURE_COLUMNS: tuple[str, ...] = (
    "agriculture_context_available",
    "agriculture_present",
    "agriculture_coverage_fraction",
    "distance_to_agriculture_km",
)

SATELLITE_COLUMNS: tuple[str, ...] = (
    "satellite_context_available",
    "satellite_source",
    "satellite_value",
    "satellite_value_name",
)

ALL_CONTEXT_COLUMNS: tuple[str, ...] = (
    LANDCOVER_COLUMNS
    + VEGETATION_COLUMNS
    + BUILTUP_COLUMNS
    + WATER_COLUMNS
    + AGRICULTURE_COLUMNS
    + SATELLITE_COLUMNS
)

I4_IMMUTABLE_COLUMNS: tuple[str, ...] = (
    "anomaly_score",
    "anomaly_status",
    "anomaly_confidence",
    "peak_frp_deviation",
    "event_size_deviation",
    "duration_deviation",
    "distance_deviation",
    "persistence_deviation",
    "monthly_deviation",
)

I5_IMMUTABLE_PREFIXES: tuple[str, ...] = ("sta_", "primary_sta_")

FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "industrial_fire",
    "wildfire",
    "agricultural_fire",
    "source_class",
    "fire_type",
    "risk_score",
    "industrial_probability",
)


def unavailable_landcover_frame(event_ids: pd.Series) -> pd.DataFrame:
    n = len(event_ids)
    return pd.DataFrame(
        {
            "event_id": event_ids.astype(str).to_numpy(),
            "landcover_available": np.full(n, False),
            "landcover_source": np.full(n, None, dtype=object),
            "landcover_year": np.full(n, None, dtype=object),
            "dominant_landcover_class": np.full(n, None, dtype=object),
            "dominant_landcover_fraction": np.full(n, np.nan),
            "landcover_class_count": np.full(n, np.nan),
        }
    )


def unavailable_binary_context_frame(event_ids: pd.Series, prefix: str) -> pd.DataFrame:
    """Build unavailable frame for vegetation/builtup/water/agriculture prefixes."""
    n = len(event_ids)
    return pd.DataFrame(
        {
            "event_id": event_ids.astype(str).to_numpy(),
            f"{prefix}_context_available": np.full(n, False),
            f"{prefix}_present": np.full(n, None, dtype=object),
            f"{prefix}_coverage_fraction": np.full(n, np.nan),
            f"distance_to_{prefix}_km": np.full(n, np.nan),
        }
    )


def unavailable_satellite_frame(event_ids: pd.Series) -> pd.DataFrame:
    n = len(event_ids)
    return pd.DataFrame(
        {
            "event_id": event_ids.astype(str).to_numpy(),
            "satellite_context_available": np.full(n, False),
            "satellite_source": np.full(n, None, dtype=object),
            "satellite_value": np.full(n, np.nan),
            "satellite_value_name": np.full(n, None, dtype=object),
        }
    )


def empty_like_unavailable(event_ids: pd.Series) -> pd.DataFrame:
    """Full unavailable context block for every event (no datasets present)."""
    frames = [
        unavailable_landcover_frame(event_ids),
        unavailable_binary_context_frame(event_ids, "vegetation"),
        unavailable_binary_context_frame(event_ids, "builtup"),
        unavailable_binary_context_frame(event_ids, "water"),
        unavailable_binary_context_frame(event_ids, "agriculture"),
        unavailable_satellite_frame(event_ids),
    ]
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="event_id", how="left")
    return out
