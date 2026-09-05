"""
End-to-end Environmental / Satellite Context pipeline (GIFT Stage I.6).

Appends context evidence without modifying G→I.5 logic or I.4/I.5 fields.
Never fabricates environmental datasets.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.environmental_context.agriculture_context import compute_agriculture_context
from src.environmental_context.builtup_context import compute_builtup_context
from src.environmental_context.config import EnvironmentalContextConfig
from src.environmental_context.context_report import build_context_report
from src.environmental_context.context_schema import (
    ALL_CONTEXT_COLUMNS,
    FORBIDDEN_SUBSTRINGS,
    I4_IMMUTABLE_COLUMNS,
    I5_IMMUTABLE_PREFIXES,
)
from src.environmental_context.landcover_context import compute_landcover_context
from src.environmental_context.satellite_context import compute_satellite_context
from src.environmental_context.vegetation_context import compute_vegetation_context
from src.environmental_context.water_context import compute_water_context


@dataclass
class ContextResult:
    events_df: pd.DataFrame
    report: dict[str, Any]


def resolve_events_input_path(config: EnvironmentalContextConfig) -> Path:
    """Prefer I.5 events CSV; fall back to I.4 when I.5 events were not produced."""
    primary = Path(config.events_path)
    if primary.exists():
        return primary
    fallback = Path(config.events_fallback_path)
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Neither I.5 events ({primary}) nor I.4 fallback ({fallback}) found."
    )


def load_events(path: str | Path) -> pd.DataFrame:
    events_path = Path(path)
    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")
    df = pd.read_csv(events_path)
    required = ("event_id", "centroid_latitude", "centroid_longitude")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"'{events_path}' missing required column(s): {missing}.")
    return df


def run_environmental_context(
    events_df: pd.DataFrame,
    config: EnvironmentalContextConfig,
    *,
    events_input_path: str = "<in-memory>",
    output_path: str = "data/processed/thermal_events_with_environmental_context.csv",
) -> ContextResult:
    """Run Stage I.6 over an already-loaded events table."""
    start = time.perf_counter()
    working = events_df.copy()
    original_ids = set(working["event_id"].astype(str))

    landcover_df, landcover_meta = compute_landcover_context(working, config)
    vegetation_df, vegetation_meta = compute_vegetation_context(working, config)
    builtup_df, builtup_meta = compute_builtup_context(working, config)
    water_df, water_meta = compute_water_context(working, config)
    agriculture_df, agriculture_meta = compute_agriculture_context(working, config)
    satellite_df, satellite_meta = compute_satellite_context(working, config)

    dataset_status = {
        "landcover": landcover_meta,
        "vegetation": vegetation_meta,
        "builtup": builtup_meta,
        "water": water_meta,
        "agriculture": agriculture_meta,
        "satellite": satellite_meta,
    }

    context = landcover_df
    for frame in (vegetation_df, builtup_df, water_df, agriculture_df, satellite_df):
        context = context.merge(frame, on="event_id", how="left")

    context = context.set_index("event_id").reindex(working["event_id"].astype(str)).reset_index()

    # Preserve all original columns; append I.6 only.
    for col in ALL_CONTEXT_COLUMNS:
        working[col] = context[col].to_numpy()

    working = working.sort_values("event_id", kind="mergesort").reset_index(drop=True)

    # Invariants
    assert len(working) == len(events_df)
    assert working["event_id"].is_unique
    assert set(working["event_id"].astype(str)) == original_ids

    original_sorted = events_df.copy()
    original_sorted["event_id"] = original_sorted["event_id"].astype(str)
    original_sorted = original_sorted.sort_values("event_id", kind="mergesort").reset_index(drop=True)
    for col in I4_IMMUTABLE_COLUMNS:
        if col not in original_sorted.columns:
            continue
        left = working[col]
        right = original_sorted[col]
        if col in ("anomaly_status", "anomaly_confidence"):
            assert left.astype(str).fillna("").tolist() == right.astype(str).fillna("").tolist()
        else:
            assert (
                pd.to_numeric(left, errors="coerce").fillna(-1e18).tolist()
                == pd.to_numeric(right, errors="coerce").fillna(-1e18).tolist()
            )

    sta_cols = [c for c in original_sorted.columns if c.startswith("sta_") or c.startswith("primary_sta")]
    for col in sta_cols:
        left = working[col]
        right = original_sorted[col]
        assert left.astype(str).fillna("").tolist() == right.astype(str).fillna("").tolist() or (
            pd.to_numeric(left, errors="coerce").fillna(-1e18).tolist()
            == pd.to_numeric(right, errors="coerce").fillna(-1e18).tolist()
        )

    cols_lower = " ".join(working.columns).lower()
    for term in ("industrial_fire", "source_class", "fire_type", "risk_score", "industrial_probability", "agricultural_fire"):
        assert term not in cols_lower
    assert "wildfire" not in cols_lower

    warnings: list[str] = []
    if not any(v.get("available") for v in dataset_status.values()):
        warnings.append(
            "No local environmental/satellite datasets were found under data/external/ "
            "or configured paths. All I.6 evidence fields are marked unavailable."
        )
    if "sta_" not in " ".join(working.columns):
        warnings.append(
            "Input events table has no I.5 STA columns (likely I.4 fallback because "
            "thermal_events_with_sta_evidence.csv was not produced). I.6 still ran."
        )

    processing_seconds = time.perf_counter() - start
    report = build_context_report(
        config=config,
        events_input_path=events_input_path,
        output_path=output_path,
        event_count=len(working),
        dataset_status=dataset_status,
        output_df=working,
        processing_seconds=processing_seconds,
        warnings=warnings,
    )
    return ContextResult(events_df=working, report=report)


def save_outputs(result: ContextResult, events_output_path: str | Path) -> None:
    path = Path(events_output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.events_df.to_csv(path, index=False)
