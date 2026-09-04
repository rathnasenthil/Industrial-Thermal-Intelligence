"""
End-to-end Geospatial Event Formation pipeline (GIFT Stage G).

Wires together, in order:

1. Load the already-cleaned FIRMS detections CSV produced by
   `src.preprocessing.run_firms_pipeline` (this stage never modifies that
   file).
2. Run ST-DBSCAN (`src.event_formation.st_dbscan`) to assign every
   detection a cluster label (or noise).
3. Aggregate clustered detections into thermal events
   (`src.event_formation.event_features`).
4. Annotate noise detections with a reason
   (`src.event_formation.noise`).
5. Assemble a reproducibility/statistics report
   (`src.event_formation.event_report`).
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.event_formation.config import STDBSCANConfig
from src.event_formation.event_features import build_thermal_events
from src.event_formation.event_report import build_event_formation_report
from src.event_formation.noise import annotate_noise
from src.event_formation.st_dbscan import run_st_dbscan

REQUIRED_DETECTION_COLUMNS: tuple[str, ...] = (
    "latitude",
    "longitude",
    "acq_datetime",
    "frp",
    "frp_valid",
    "bright_ti4",
    "bright_ti5",
    "confidence",
    "daynight",
)


@dataclass
class EventFormationResult:
    """Result of running the full Stage G pipeline.

    Attributes:
        events_df: One row per thermal event.
        detections_df: All clustered (non-noise) detections, each with an
            `event_id` column.
        noise_df: All unclustered detections, each with
            `spatiotemporal_neighbor_count` and `noise_reason` columns.
        report: JSON-serializable event-formation report.
    """

    events_df: pd.DataFrame
    detections_df: pd.DataFrame
    noise_df: pd.DataFrame
    report: dict[str, Any]


def load_clean_detections(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load the cleaned FIRMS detections CSV produced by preprocessing.

    Args:
        path: Path to `firms_viirs_india_2023_2024_clean.csv` (or a
            compatible file with the same schema).
        max_rows: If given, only the first `max_rows` rows are loaded
            (useful for benchmarking on a representative subset).

    Returns:
        A DataFrame with `acq_datetime` parsed as timezone-aware (UTC)
        timestamps.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If required columns are missing.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Cleaned FIRMS detections file not found: {csv_path}")

    df = pd.read_csv(csv_path, nrows=max_rows)
    missing = [c for c in REQUIRED_DETECTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Cleaned FIRMS detections file '{csv_path}' is missing required "
            f"column(s): {missing}. This pipeline expects the output of "
            f"src.preprocessing.run_firms_pipeline."
        )

    df["acq_datetime"] = pd.to_datetime(df["acq_datetime"], utc=True)
    return df


def _assign_event_ids(labels: np.ndarray, acq_datetime: pd.Series) -> np.ndarray:
    """Map raw dense cluster labels to stable, human-readable event ids.

    Clusters are ordered by their earliest detection timestamp so that
    `event_id` sequence numbers roughly follow chronological order; this
    is purely cosmetic and has no effect on clustering itself.

    Args:
        labels: Raw ST-DBSCAN labels (``-1`` = noise).
        acq_datetime: Acquisition timestamps aligned with `labels`.

    Returns:
        An object array of the same length as `labels`: an
        ``"EVT_0000001"``-style string for clustered points, or ``None``
        for noise points.
    """
    event_ids: np.ndarray = np.full(len(labels), None, dtype=object)
    clustered_mask = labels >= 0
    if not np.any(clustered_mask):
        return event_ids

    cluster_ids = np.unique(labels[clustered_mask])
    earliest = {
        cid: acq_datetime[labels == cid].min() for cid in cluster_ids
    }
    ordered = sorted(cluster_ids, key=lambda cid: earliest[cid])
    id_map = {cid: f"EVT_{seq + 1:07d}" for seq, cid in enumerate(ordered)}

    for cid in cluster_ids:
        event_ids[labels == cid] = id_map[cid]
    return event_ids


def run_event_formation(
    detections: pd.DataFrame,
    config: STDBSCANConfig,
    input_path: str = "<in-memory>",
    measure_memory: bool = True,
) -> EventFormationResult:
    """Run the full Stage G pipeline over an already-loaded detections DataFrame.

    Args:
        detections: Output of `load_clean_detections` (or an equivalent
            DataFrame with the same required columns).
        config: ST-DBSCAN clustering configuration.
        input_path: Recorded in the report for provenance only.
        measure_memory: Whether to track peak memory via `tracemalloc`
            (adds modest overhead; disable for repeated benchmark runs if
            undesired).

    Returns:
        An `EventFormationResult`.
    """
    detections = detections.reset_index(drop=True)

    if measure_memory:
        tracemalloc.start()
    start_time = time.perf_counter()

    cluster_result = run_st_dbscan(
        detections["latitude"].to_numpy(dtype=np.float64),
        detections["longitude"].to_numpy(dtype=np.float64),
        detections["acq_datetime"],
        config,
    )

    event_ids = _assign_event_ids(cluster_result.labels, detections["acq_datetime"])
    neighbor_counts = pd.Series(cluster_result.neighbor_counts, index=detections.index)

    detections = detections.copy()
    detections["event_id"] = event_ids

    clustered_mask = cluster_result.labels >= 0
    detections_df = detections.loc[clustered_mask].reset_index(drop=True)
    noise_df_raw = detections.loc[~clustered_mask].drop(columns=["event_id"])

    events_df = build_thermal_events(detections_df) if not detections_df.empty else pd.DataFrame()
    noise_df = annotate_noise(noise_df_raw, neighbor_counts, config)

    processing_seconds = time.perf_counter() - start_time
    peak_memory_mb = None
    if measure_memory:
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_memory_mb = peak / (1024 * 1024)

    report = build_event_formation_report(
        config=config,
        input_detection_count=len(detections),
        events_df=events_df,
        noise_df=noise_df,
        processing_seconds=processing_seconds,
        peak_memory_mb=peak_memory_mb,
        input_path=str(input_path),
    )

    return EventFormationResult(
        events_df=events_df, detections_df=detections_df, noise_df=noise_df, report=report
    )


def save_outputs(
    result: EventFormationResult,
    events_path: str | Path,
    detections_path: str | Path,
    noise_path: str | Path,
) -> None:
    """Write the events, detections and noise tables to CSV.

    Args:
        result: Output of `run_event_formation`.
        events_path: Destination for `thermal_events.csv`.
        detections_path: Destination for `thermal_event_detections.csv`.
        noise_path: Destination for `thermal_event_noise.csv`.
    """
    for path, df in (
        (events_path, result.events_df),
        (detections_path, result.detections_df),
        (noise_path, result.noise_df),
    ):
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
