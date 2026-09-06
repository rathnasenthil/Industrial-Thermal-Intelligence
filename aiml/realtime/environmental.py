"""
Incremental Stage I.6 Environmental / Satellite Context (AIML realtime adapter).

Environmental context is **context/evidence only** — not ground truth,
industrial-fire classification, anomaly labels, or risk scoring.

Why realtime I.6 cannot replay the full batch table
------------------------------------------------------
Batch ``run_environmental_context()`` appends I.6 columns for every event in
the Stage I.5 (or I.4 fallback) table. On each NRT poll only the affected
event needs updating.

This adapter calls the same batch pipeline on a **one-event** frame so
landcover / vector-presence / satellite sampling rules stay identical.

When local environmental datasets under ``data/external/`` (or configured
paths) are absent, returns the same unavailable defaults as batch
``empty_like_unavailable`` — never fabricates environmental values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import pandas as pd

from src.environmental_context.config import DEFAULT_CONFIG, EnvironmentalContextConfig
from src.environmental_context.context_pipeline import run_environmental_context
from src.environmental_context.context_schema import (
    ALL_CONTEXT_COLUMNS,
    empty_like_unavailable,
)
from src.environmental_context.raster_loader import resolve_existing_path

REQUIRED_EVENT_COLUMNS: tuple[str, ...] = (
    "event_id",
    "centroid_latitude",
    "centroid_longitude",
)


def any_environmental_source_present(config: EnvironmentalContextConfig) -> bool:
    """True if at least one configured local I.6 source file exists on disk."""
    for path in (
        config.landcover_raster_path,
        config.landcover_vector_path,
        config.vegetation_path,
        config.builtup_path,
        config.water_path,
        config.agriculture_path,
        config.satellite_raster_path,
    ):
        if resolve_existing_path(path) is not None:
            return True
    return False


@dataclass(frozen=True)
class EnvironmentalContextResult:
    """I.6 fields for one event (batch ``ALL_CONTEXT_COLUMNS``)."""

    event_id: str
    landcover_available: bool
    landcover_source: str | None
    landcover_year: str | None
    dominant_landcover_class: str | None
    dominant_landcover_fraction: float | None
    landcover_class_count: float | None
    vegetation_context_available: bool
    vegetation_present: bool | None
    vegetation_coverage_fraction: float | None
    distance_to_vegetation_km: float | None
    builtup_context_available: bool
    builtup_present: bool | None
    builtup_coverage_fraction: float | None
    distance_to_builtup_km: float | None
    water_context_available: bool
    water_present: bool | None
    water_coverage_fraction: float | None
    distance_to_water_km: float | None
    agriculture_context_available: bool
    agriculture_present: bool | None
    agriculture_coverage_fraction: float | None
    distance_to_agriculture_km: float | None
    satellite_context_available: bool
    satellite_source: str | None
    satellite_value: float | None
    satellite_value_name: str | None
    source_missing: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def unavailable_environmental_result(
    event_id: str, *, source_missing: bool = False
) -> EnvironmentalContextResult:
    """Batch-equivalent defaults when all I.6 sources are missing."""
    frame = empty_like_unavailable(pd.Series([str(event_id)]))
    return _row_to_result(frame.iloc[0], source_missing=source_missing)


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


def _opt_bool(val: Any) -> bool | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (bool, int)):
        return bool(val)
    text = str(val).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    return None


def _req_bool(val: Any, default: bool = False) -> bool:
    parsed = _opt_bool(val)
    return default if parsed is None else parsed


def _row_to_result(row: pd.Series, *, source_missing: bool = False) -> EnvironmentalContextResult:
    return EnvironmentalContextResult(
        event_id=str(row["event_id"]),
        landcover_available=_req_bool(row.get("landcover_available"), False),
        landcover_source=_opt_str(row.get("landcover_source")),
        landcover_year=_opt_str(row.get("landcover_year")),
        dominant_landcover_class=_opt_str(row.get("dominant_landcover_class")),
        dominant_landcover_fraction=_opt_float(row.get("dominant_landcover_fraction")),
        landcover_class_count=_opt_float(row.get("landcover_class_count")),
        vegetation_context_available=_req_bool(row.get("vegetation_context_available"), False),
        vegetation_present=_opt_bool(row.get("vegetation_present")),
        vegetation_coverage_fraction=_opt_float(row.get("vegetation_coverage_fraction")),
        distance_to_vegetation_km=_opt_float(row.get("distance_to_vegetation_km")),
        builtup_context_available=_req_bool(row.get("builtup_context_available"), False),
        builtup_present=_opt_bool(row.get("builtup_present")),
        builtup_coverage_fraction=_opt_float(row.get("builtup_coverage_fraction")),
        distance_to_builtup_km=_opt_float(row.get("distance_to_builtup_km")),
        water_context_available=_req_bool(row.get("water_context_available"), False),
        water_present=_opt_bool(row.get("water_present")),
        water_coverage_fraction=_opt_float(row.get("water_coverage_fraction")),
        distance_to_water_km=_opt_float(row.get("distance_to_water_km")),
        agriculture_context_available=_req_bool(row.get("agriculture_context_available"), False),
        agriculture_present=_opt_bool(row.get("agriculture_present")),
        agriculture_coverage_fraction=_opt_float(row.get("agriculture_coverage_fraction")),
        distance_to_agriculture_km=_opt_float(row.get("distance_to_agriculture_km")),
        satellite_context_available=_req_bool(row.get("satellite_context_available"), False),
        satellite_source=_opt_str(row.get("satellite_source")),
        satellite_value=_opt_float(row.get("satellite_value")),
        satellite_value_name=_opt_str(row.get("satellite_value_name")),
        source_missing=source_missing,
    )


def process_event_environmental(
    events_df: pd.DataFrame,
    event_id: str,
    *,
    config: Optional[EnvironmentalContextConfig] = None,
) -> EnvironmentalContextResult:
    """
    Compute I.6 environmental context for **one** event using batch semantics.

    Args:
        events_df: Must contain the current event row with centroid columns
            expected by batch I.6.
        event_id: Event to score.
        config: Defaults to batch ``DEFAULT_CONFIG``.

    Returns:
        EnvironmentalContextResult for ``event_id`` only.
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

    work = events_df.loc[events_df["event_id"].astype(str) == eid].copy()
    if len(work) != 1:
        raise ValueError(f"Expected exactly one row for event_id={eid}, got {len(work)}.")

    if not any_environmental_source_present(cfg):
        return unavailable_environmental_result(eid, source_missing=True)

    result = run_environmental_context(work, cfg)
    for col in ALL_CONTEXT_COLUMNS:
        if col not in result.events_df.columns:
            raise RuntimeError(f"Batch I.6 output missing column: {col}")

    row = result.events_df.loc[result.events_df["event_id"].astype(str) == eid].iloc[0]
    # source_missing only when *no* local datasets existed; partial availability
    # still reports source_missing=False (batch can mark per-domain unavailable).
    return _row_to_result(row, source_missing=False)
