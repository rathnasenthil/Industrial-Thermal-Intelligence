"""
Event-formation report assembly for GIFT Stage G.

Builds a single JSON-serializable summary of a clustering run: row
counts, event-size/duration/FRP statistics, processing time, and the
exact clustering configuration used — enough to reproduce the run
(clustering is deterministic given the same input and config; see
`src.event_formation.st_dbscan`).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.event_formation.config import STDBSCANConfig


def _quantile_stats(values: pd.Series) -> dict[str, float | None]:
    if values.empty:
        return {"min": None, "median": None, "max": None, "mean": None}
    return {
        "min": float(values.min()),
        "median": float(values.median()),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def build_event_formation_report(
    *,
    config: STDBSCANConfig,
    input_detection_count: int,
    events_df: pd.DataFrame,
    noise_df: pd.DataFrame,
    processing_seconds: float,
    peak_memory_mb: float | None,
    input_path: str,
) -> dict[str, Any]:
    """Assemble the full event-formation report as a JSON-safe dict.

    Args:
        config: The `STDBSCANConfig` used for this run.
        input_detection_count: Number of input (cleaned) FIRMS detections
            fed into clustering.
        events_df: The final `thermal_events` DataFrame (one row per
            event).
        noise_df: The final `thermal_event_noise` DataFrame.
        processing_seconds: Wall-clock seconds spent on clustering + event
            aggregation.
        peak_memory_mb: Peak memory (MB) observed via `tracemalloc` during
            clustering, if measured (``None`` if not measured).
        input_path: Path to the input detections CSV (for provenance).

    Returns:
        A JSON-serializable dict.
    """
    n_events = len(events_df)
    n_noise = len(noise_df)
    n_clustered = input_detection_count - n_noise

    pct_clustered = (n_clustered / input_detection_count * 100.0) if input_detection_count else 0.0
    pct_noise = (n_noise / input_detection_count * 100.0) if input_detection_count else 0.0

    if n_events:
        event_sizes = events_df["detection_count"]
        durations = events_df["observed_duration_hours"]
        peak_frps = events_df["peak_frp"].dropna()
    else:
        event_sizes = pd.Series(dtype=float)
        durations = pd.Series(dtype=float)
        peak_frps = pd.Series(dtype=float)

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_stage": "GIFT Stage G - Geospatial Event Formation (ST-DBSCAN)",
        "input": {
            "path": input_path,
            "detection_count": int(input_detection_count),
        },
        "clustering_algorithm": (
            "ST-DBSCAN (Birant & Kut, 2007 formulation): DBSCAN with a joint "
            "spatial (great-circle/haversine) + temporal neighborhood "
            "definition. Implemented via a batched BallTree(haversine) "
            "radius-neighbor graph fed into sklearn.cluster.DBSCAN "
            "(metric='precomputed'). Not a novel algorithm; see "
            "src/event_formation/st_dbscan.py for engineering details."
        ),
        "clustering_config": {
            **config.to_dict(),
            "parameter_rationale": config.describe_rationale(),
            "parameters_are_scientifically_validated": False,
        },
        "counts": {
            "input_detection_count": int(input_detection_count),
            "event_count": int(n_events),
            "clustered_detection_count": int(n_clustered),
            "noise_detection_count": int(n_noise),
            "percent_clustered": round(pct_clustered, 4),
            "percent_noise": round(pct_noise, 4),
        },
        "event_size_stats": {
            **_quantile_stats(event_sizes),
            "total_events": int(n_events),
        },
        "event_duration_hours_stats": _quantile_stats(durations),
        "event_peak_frp_stats": _quantile_stats(peak_frps),
        "performance": {
            "processing_seconds": round(processing_seconds, 3),
            "peak_memory_mb": (round(peak_memory_mb, 2) if peak_memory_mb is not None else None),
            "memory_conscious_design_notes": [
                "Spatial neighbor queries are batched (config.query_batch_size "
                "detections per batch) against a single global BallTree so "
                "peak memory holds only one batch's neighbor-index arrays at "
                "a time, not all detections' neighbor lists simultaneously.",
                "No O(n^2) dense distance matrix is ever constructed; the "
                "spatio-temporal neighbor graph is stored sparsely (only "
                "true neighbor pairs).",
            ],
        },
        "reproducibility": {
            "deterministic": True,
            "notes": (
                "Given the same input CSV (same row order) and the same "
                "STDBSCANConfig values recorded in 'clustering_config' above, "
                "re-running src.event_formation.run_event_formation "
                "reproduces identical event_id assignments and statistics. "
                "No random sampling or randomized algorithm is used."
            ),
        },
        "notes": [
            "This report covers GIFT Stage G (Geospatial Event Formation) "
            "only: no OSM/NASA-STA/Sentinel-2 context, facility "
            "fingerprinting, anomaly detection, ML classification, or risk "
            "scoring is performed at this stage.",
            "Clusters produced here are called 'thermal events', never "
            "'fires' — a FIRMS detection is a satellite-observed thermal "
            "anomaly whose source has not yet been classified.",
            "'observed_duration_hours' reflects only the span between the "
            "first and last *observed* satellite detection in an event; it "
            "is not necessarily the true physical duration of the "
            "underlying thermal source (satellites do not observe "
            "continuously, and no interpolation between overpasses is "
            "performed).",
            "Noise (unclustered) detections are preserved in "
            "thermal_event_noise.csv, never deleted.",
        ],
    }
    return report


def save_report(report: dict[str, Any], path: str | Path) -> None:
    """Write the report dict to disk as pretty-printed JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=False, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")
