"""
Incremental Stage I.5 NASA STA evidence (AIML realtime adapter).

STA is **supporting evidence only** — not ground truth, not industrial-fire
classification, and not risk scoring.

Why realtime I.5 cannot replay the full batch pipeline
------------------------------------------------------
Batch ``run_sta_integration()`` appends I.5 columns for every event in the
Stage I.4 table. On each NRT poll only the affected event needs updating.

This adapter calls the same batch pipeline on a **one-event** frame so
matching, ranking, ambiguity, and quality rules stay identical.

When no local NASA STA Mask/Detections file exists, returns the same
NO_STA_ASSOCIATION defaults as an event with zero candidates — never
fabricates STA geometries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import pandas as pd

from src.sta_evidence.config import (
    DEFAULT_CONFIG,
    NO_STA_ASSOCIATION,
    QUALITY_NONE,
    STAConfig,
)
from src.sta_evidence.sta_loader import STASourceMissingError, resolve_existing_paths
from src.sta_evidence.sta_pipeline import I5_APPEND_COLUMNS, run_sta_integration

REQUIRED_EVENT_COLUMNS: tuple[str, ...] = (
    "event_id",
    "event_start",
    "event_end",
    "footprint_wkt",
    "centroid_latitude",
    "centroid_longitude",
)


@dataclass(frozen=True)
class STAEvidenceResult:
    """I.5 fields for one event."""

    event_id: str
    sta_association_status: str
    primary_sta_id: str | None
    sta_layer_type: str | None
    sta_match_count: int
    sta_nearest_distance_km: float | None
    sta_intersection_area_m2: float | None
    sta_evidence_available: bool
    sta_temporal_relation: str | None
    sta_evidence_quality: str
    source_missing: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def unavailable_sta_result(event_id: str, *, source_missing: bool = False) -> STAEvidenceResult:
    """Batch-equivalent defaults when STA source is missing or no candidates."""
    return STAEvidenceResult(
        event_id=str(event_id),
        sta_association_status=NO_STA_ASSOCIATION,
        primary_sta_id=None,
        sta_layer_type=None,
        sta_match_count=0,
        sta_nearest_distance_km=None,
        sta_intersection_area_m2=None,
        sta_evidence_available=False,
        sta_temporal_relation=None,
        sta_evidence_quality=QUALITY_NONE,
        source_missing=source_missing,
    )


def _row_to_result(row: pd.Series, *, source_missing: bool = False) -> STAEvidenceResult:
    def _opt_float(val: Any) -> float | None:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        try:
            f = float(val)
        except (TypeError, ValueError):
            return None
        if pd.isna(f):
            return None
        return f

    def _opt_str(val: Any) -> str | None:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        text = str(val).strip()
        if not text or text.lower() == "nan":
            return None
        return text

    available = row.get("sta_evidence_available")
    if available is None or (isinstance(available, float) and pd.isna(available)):
        available = False
    else:
        available = bool(available)

    match_count = row.get("sta_match_count")
    try:
        match_count_i = int(match_count) if match_count is not None and not pd.isna(match_count) else 0
    except (TypeError, ValueError):
        match_count_i = 0

    return STAEvidenceResult(
        event_id=str(row["event_id"]),
        sta_association_status=str(row.get("sta_association_status") or NO_STA_ASSOCIATION),
        primary_sta_id=_opt_str(row.get("primary_sta_id")),
        sta_layer_type=_opt_str(row.get("sta_layer_type")),
        sta_match_count=match_count_i,
        sta_nearest_distance_km=_opt_float(row.get("sta_nearest_distance_km")),
        sta_intersection_area_m2=_opt_float(row.get("sta_intersection_area_m2")),
        sta_evidence_available=available,
        sta_temporal_relation=_opt_str(row.get("sta_temporal_relation")),
        sta_evidence_quality=str(row.get("sta_evidence_quality") or QUALITY_NONE),
        source_missing=source_missing,
    )


def process_event_sta(
    events_df: pd.DataFrame,
    event_id: str,
    *,
    config: Optional[STAConfig] = None,
    sta_gdf=None,
) -> STAEvidenceResult:
    """
    Compute I.5 STA evidence for **one** event using batch semantics.

    Args:
        events_df: Must contain the current event row with geometry columns
            expected by batch I.5 (footprint/centroid).
        event_id: Event to score.
        config: Defaults to batch ``DEFAULT_CONFIG``.
        sta_gdf: Optional pre-loaded STA GeoDataFrame (tests / cached load).
            When None, attempts to load from ``config`` paths.

    Returns:
        STAEvidenceResult for ``event_id`` only.
    """
    cfg = config or DEFAULT_CONFIG
    eid = str(event_id)

    if events_df is None or events_df.empty:
        raise ValueError("events_df must contain the current event.")
    for col in REQUIRED_EVENT_COLUMNS:
        if col not in events_df.columns:
            raise ValueError(f"Events dataframe missing required column: {col}")
    if not (events_df["event_id"].astype(str) == eid).any():
        raise ValueError(f"event_id={eid} not present in events_df.")

    # Single-event frame for incremental processing.
    work = events_df.loc[events_df["event_id"].astype(str) == eid].copy()
    if len(work) != 1:
        raise ValueError(f"Expected exactly one row for event_id={eid}, got {len(work)}.")

    if sta_gdf is None and not resolve_existing_paths(cfg):
        return unavailable_sta_result(eid, source_missing=True)

    try:
        result = run_sta_integration(work, cfg, sta_gdf=sta_gdf)
    except STASourceMissingError:
        return unavailable_sta_result(eid, source_missing=True)

    row = result.events_df.loc[result.events_df["event_id"].astype(str) == eid].iloc[0]
    # Ensure I.5 columns present
    for col in I5_APPEND_COLUMNS:
        if col not in result.events_df.columns:
            raise RuntimeError(f"Batch I.5 output missing column: {col}")
    return _row_to_result(row, source_missing=False)
