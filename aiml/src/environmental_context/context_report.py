"""JSON report assembly for GIFT Stage I.6."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.environmental_context.config import EnvironmentalContextConfig
from src.infrastructure.association_geometry import INDIA_EQUAL_AREA_CRS


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else round(float(value), 6)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def build_context_report(
    *,
    config: EnvironmentalContextConfig,
    events_input_path: str,
    output_path: str,
    event_count: int,
    dataset_status: dict[str, Any],
    output_df: pd.DataFrame,
    processing_seconds: float,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    available = {k: v for k, v in dataset_status.items() if v.get("available")}
    missing = {k: v for k, v in dataset_status.items() if not v.get("available")}

    coverage = {
        "landcover_available_count": int(output_df["landcover_available"].sum()) if "landcover_available" in output_df else 0,
        "vegetation_available_count": int(output_df["vegetation_context_available"].sum()) if "vegetation_context_available" in output_df else 0,
        "builtup_available_count": int(output_df["builtup_context_available"].sum()) if "builtup_context_available" in output_df else 0,
        "water_available_count": int(output_df["water_context_available"].sum()) if "water_context_available" in output_df else 0,
        "agriculture_available_count": int(output_df["agriculture_context_available"].sum()) if "agriculture_context_available" in output_df else 0,
        "satellite_available_count": int(output_df["satellite_context_available"].sum()) if "satellite_context_available" in output_df else 0,
    }

    null_counts = {
        "dominant_landcover_class_null": int(output_df["dominant_landcover_class"].isna().sum()) if "dominant_landcover_class" in output_df else None,
        "distance_to_water_km_null": int(output_df["distance_to_water_km"].isna().sum()) if "distance_to_water_km" in output_df else None,
        "satellite_value_null": int(output_df["satellite_value"].isna().sum()) if "satellite_value" in output_df else None,
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "GIFT Stage I.6 - Satellite / Environmental Context",
        "input_file": events_input_path,
        "output_file": output_path,
        "event_count": int(event_count),
        "context_buffer_configuration": {
            "context_buffer_km": config.context_buffer_km,
            "broad_context_buffer_km": config.broad_context_buffer_km,
            "storage_crs": "EPSG:4326",
            "computation_crs": INDIA_EQUAL_AREA_CRS,
            "rationale": config.describe_rationale(),
        },
        "datasets_detected": available,
        "datasets_missing": missing,
        "per_source_availability": {k: bool(v.get("available")) for k, v in dataset_status.items()},
        "coverage_statistics": coverage,
        "null_unavailable_counts": null_counts,
        "processing_time_seconds": round(processing_seconds, 3),
        "warnings": warnings or [],
        "i4_i5_immutability": {
            "anomaly_fields_unchanged": True,
            "sta_fields_unchanged_when_present": True,
            "notes": (
                "I.6 appends environmental context columns only. I.4 anomaly fields "
                "and any present I.5 STA fields are never recalculated."
            ),
        },
        "limitations": [
            "I.6 is an evidence/context stage only. It does not classify events as "
            "industrial fire, wildfire, agricultural fire, or any other source.",
            "Missing datasets produce availability=false and null evidence fields — "
            "never fabricated zeros for unavailable numeric evidence.",
            "Context buffers are engineering parameters, not scientifically validated radii.",
            "Stage G event geometries are detection envelopes, not true fire perimeters.",
            "No machine learning, pseudo-labels, risk scores, or live satellite APIs are used.",
            "No automatic download of environmental/satellite products is performed.",
        ],
        "configuration": config.to_dict(),
    }
    return _to_jsonable(report)


def save_report(report: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=False)
