"""
Report assembly for GIFT Stage I.2 (Thermal Event <-> Facility Association).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.infrastructure.association_config import AssociationConfig
from src.infrastructure.association_geometry import INDIA_EQUAL_AREA_CRS
from src.infrastructure.facility_association import (
    AMBIGUOUS,
    CONFIDENCE_LEVELS,
    INTERSECTS_FACILITY,
    NEAR_FACILITY,
    NO_FACILITY_ASSOCIATION,
    WITHIN_FACILITY,
)
from src.infrastructure.facility_schema import FACILITY_TYPES


def _distance_stats(series: pd.Series) -> dict[str, float | None]:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return {"min_km": None, "median_km": None, "mean_km": None, "max_km": None}
    return {
        "min_km": round(float(valid.min()), 3),
        "median_km": round(float(valid.median()), 3),
        "mean_km": round(float(valid.mean()), 3),
        "max_km": round(float(valid.max()), 3),
    }


_CANDIDATE_COUNT_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("0", 0, 0),
    ("1", 1, 1),
    ("2-5", 2, 5),
    ("6-10", 6, 10),
    ("11-50", 11, 50),
    ("51-200", 51, 200),
    ("201+", 201, 10**9),
)


def _candidate_count_summary(candidate_counts: pd.Series) -> dict[str, Any]:
    """Bucketed summary of `candidate_facility_count` (avoids a per-value
    dict with hundreds of keys -- real industrial clusters can have
    candidate counts in the thousands, see report limitations)."""
    buckets = {}
    for label, lo, hi in _CANDIDATE_COUNT_BUCKETS:
        buckets[label] = int(((candidate_counts >= lo) & (candidate_counts <= hi)).sum())
    nonzero = candidate_counts[candidate_counts > 0]
    return {
        "bucketed_event_counts": buckets,
        "min": int(candidate_counts.min()) if len(candidate_counts) else None,
        "median_excluding_zero": (float(nonzero.median()) if not nonzero.empty else None),
        "mean_excluding_zero": (round(float(nonzero.mean()), 2) if not nonzero.empty else None),
        "max": int(candidate_counts.max()) if len(candidate_counts) else None,
    }


def build_association_report(
    *,
    config: AssociationConfig,
    events_input_path: str,
    facilities_input_path: str,
    event_count: int,
    facility_count: int,
    events_with_association_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    processing_seconds: float,
) -> dict[str, Any]:
    """Assemble the Stage I.2 report as a JSON-serializable dict.

    Args:
        config: The `AssociationConfig` used for this run.
        events_input_path: Path to the event table that was read.
        facilities_input_path: Path to the facility table that was read.
        event_count: Number of input events (before this stage's merge).
        facility_count: Number of input facilities.
        events_with_association_df: Output of
            `association_pipeline.run_facility_association` (events_df).
        candidates_df: Output of `association_pipeline.run_facility_association`
            (candidates_df).
        processing_seconds: Wall-clock seconds spent on the full pipeline.

    Returns:
        A JSON-serializable dict.
    """
    method_counts = events_with_association_df["facility_association_method"].value_counts()
    confidence_counts = events_with_association_df["facility_attribution_confidence"].value_counts()
    facility_type_counts = events_with_association_df["facility_type"].value_counts(dropna=True)

    events_with_facility = int((events_with_association_df["facility_id"].notna()).sum())
    events_without_facility = int(event_count - events_with_facility)

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_stage": "GIFT Stage I.2 - Thermal Event <-> Facility Association",
        "input": {
            "event_input_path": events_input_path,
            "facility_input_path": facilities_input_path,
            "event_count": int(event_count),
            "facility_count": int(facility_count),
        },
        "association_results": {
            "events_with_facility_association": events_with_facility,
            "events_without_facility_association": events_without_facility,
            "events_within_facility": int(method_counts.get(WITHIN_FACILITY, 0)),
            "events_intersecting_facility": int(method_counts.get(INTERSECTS_FACILITY, 0)),
            "events_near_facility": int(method_counts.get(NEAR_FACILITY, 0)),
            "events_ambiguous": int(method_counts.get(AMBIGUOUS, 0)),
            "events_no_association": int(method_counts.get(NO_FACILITY_ASSOCIATION, 0)),
        },
        "facility_type_counts": {t: int(facility_type_counts.get(t, 0)) for t in FACILITY_TYPES},
        "confidence_counts": {c: int(confidence_counts.get(c, 0)) for c in CONFIDENCE_LEVELS},
        "distance_statistics_km": _distance_stats(events_with_association_df["facility_distance_km"]),
        "candidate_statistics": {
            **_candidate_count_summary(events_with_association_df["candidate_facility_count"]),
            "total_candidate_pairs_recorded": int(len(candidates_df)),
            "max_candidates_per_event_cap": config.max_candidates_per_event,
        },
        "performance": {"processing_seconds": round(processing_seconds, 3)},
        "configuration": {
            **config.to_dict(),
            "crs_strategy": {
                "persisted_crs": "EPSG:4326 (WGS84, matches Stage G / Stage I.1 outputs)",
                "computation_crs": INDIA_EQUAL_AREA_CRS,
                "computation_crs_description": (
                    "Custom India-centered Albers Equal-Area Conic (standard "
                    "parallels 8N/37N, central meridian 82E), used only "
                    "internally for buffering/distance/containment "
                    "calculations -- see association_geometry.py module "
                    "docstring for full rationale. Never persisted to output "
                    "files, which remain EPSG:4326 throughout."
                ),
            },
        },
        "reproducibility": {
            "deterministic": True,
            "notes": (
                "Given the same event table, facility table and configuration, "
                "candidate ranking, selection and all statistics above are "
                "identical across repeated runs (see facility_association.py "
                "for the explicit deterministic tie-breaking chain)."
            ),
        },
        "limitations": [
            "FACILITY ASSOCIATION IS NOT SOURCE CLASSIFICATION: a "
            "WITHIN_FACILITY or NEAR_FACILITY result means the event is "
            "spatially plausible near a facility -- it is NOT proof that "
            "the facility caused the thermal event. No field in this "
            "stage's output should be read as industrial/wildfire/"
            "agricultural source classification; that is a later, "
            "not-yet-implemented GIFT stage.",
            "OSM is contextual, crowd-sourced evidence, not a ground-truth "
            "industrial-activity registry (see Stage I.1). OSM coverage is "
            "known to be incomplete and inconsistent, especially in parts "
            "of India -- NO_FACILITY_ASSOCIATION means no suitable OSM "
            "record was found near the event, NOT that the event is "
            "confirmed non-industrial.",
            "Facility geometry quality varies: some facilities are mapped "
            "as precise polygons, others only as a single representative "
            "node/point, and Stage I.1 does not reconstruct multipolygon "
            "relation geometry -- see osm_facility_report.json for exact "
            "coverage/rejection counts.",
            f"association_radius_km ({config.association_radius_km}) is an "
            "engineering search-radius threshold, not a scientifically "
            "validated or causally calibrated distance -- see "
            "association_config.py for full rationale.",
            "AMBIGUOUS events intentionally have no single facility_id "
            "selected in the main output, even though candidate facilities "
            "exist -- see thermal_event_facility_candidates.csv for the "
            "full, ranked candidate list. This is deliberate: the pipeline "
            "never blindly selects the nearest facility when multiple "
            "candidates cannot be confidently distinguished.",
            "Event geometry (footprint_wkt) is the Stage G observed-"
            "detection envelope, not the true physical fire/thermal-source "
            "perimeter -- see src.event_formation.geometry. Containment/"
            "intersection results inherit that same limitation.",
        ],
    }
    return _to_jsonable(report)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def save_report(report: dict[str, Any], path: str | Path) -> None:
    """Write the report dict to disk as pretty-printed JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=False)
