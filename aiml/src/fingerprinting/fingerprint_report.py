"""Report assembly for GIFT Stage I.3 (Facility Fingerprinting)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.fingerprinting.fingerprint_config import (
    ESTABLISHED_BASELINE,
    FingerprintConfig,
    INSUFFICIENT_HISTORY,
    LIMITED_HISTORY,
    NO_OBSERVATIONS,
)
from src.infrastructure.facility_schema import FACILITY_TYPES


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
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    return value


def build_fingerprint_report(
    *,
    config: FingerprintConfig,
    events_input_path: str,
    facilities_input_path: str,
    facility_count: int,
    event_count: int,
    events_with_association_df: pd.DataFrame,
    fingerprints_df: pd.DataFrame,
    processing_seconds: float,
) -> dict[str, Any]:
    """Assemble the Stage I.3 report as a JSON-serializable dict.

    Args:
        config: The `FingerprintConfig` used for this run.
        events_input_path: Path to the Stage I.2 event/association table read.
        facilities_input_path: Path to the Stage I.1 facility table read.
        facility_count: Number of facilities in the input facility universe.
        event_count: Number of input events (all of Stage G/G.1, not just associated).
        events_with_association_df: The Stage I.2 table (unmodified) --
            used to compute the confirmed-associated-event-only statistics
            (persistence distribution, confidence composition, temporal
            coverage).
        fingerprints_df: Output of `build_facility_fingerprints`.
        processing_seconds: Wall-clock seconds spent on the full pipeline.

    Returns:
        A JSON-serializable dict.
    """
    associated = events_with_association_df.loc[events_with_association_df["facility_id"].notna()]

    status_counts = fingerprints_df["fingerprint_status"].value_counts()
    facilities_with_observations = int((fingerprints_df["event_count"] > 0).sum())

    observed = fingerprints_df.loc[fingerprints_df["event_count"] > 0, "event_count"]
    observation_stats = (
        {
            "min_events_per_facility": int(observed.min()),
            "median_events_per_facility": float(observed.median()),
            "mean_events_per_facility": round(float(observed.mean()), 3),
            "max_events_per_facility": int(observed.max()),
        }
        if not observed.empty
        else {
            "min_events_per_facility": None,
            "median_events_per_facility": None,
            "mean_events_per_facility": None,
            "max_events_per_facility": None,
        }
    )

    persistence_dist = associated["persistence_label"].value_counts() if not associated.empty else pd.Series(dtype="int64")
    confidence_dist = (
        associated["facility_attribution_confidence"].value_counts() if not associated.empty else pd.Series(dtype="int64")
    )
    type_dist = (
        fingerprints_df.loc[fingerprints_df["event_count"] > 0, "facility_type"].value_counts()
        if facilities_with_observations
        else pd.Series(dtype="int64")
    )

    first_obs = pd.to_datetime(associated["event_start"], utc=True).min() if not associated.empty else None
    last_obs = pd.to_datetime(associated["event_end"], utc=True).max() if not associated.empty else None

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_stage": "GIFT Stage I.3 - Facility Fingerprinting & Historical Thermal Baseline",
        "input": {
            "events_input_path": events_input_path,
            "facilities_input_path": facilities_input_path,
            "facility_count": int(facility_count),
            "event_count": int(event_count),
            "associated_event_count": int(len(associated)),
        },
        "fingerprint_coverage": {
            "facilities_with_observations": facilities_with_observations,
            "facilities_without_observations": int(status_counts.get(NO_OBSERVATIONS, 0)),
            "facilities_with_insufficient_history": int(status_counts.get(INSUFFICIENT_HISTORY, 0)),
            "facilities_with_limited_history": int(status_counts.get(LIMITED_HISTORY, 0)),
            "facilities_with_established_baseline": int(status_counts.get(ESTABLISHED_BASELINE, 0)),
        },
        "observation_statistics": {
            **observation_stats,
            "note": "Computed only over facilities with >=1 confirmed observation; including the majority of facilities with zero observations would trivially collapse every statistic toward zero.",
        },
        "persistence_distribution_among_confirmed_events": {
            "PERSISTENT": int(persistence_dist.get("PERSISTENT", 0)),
            "RECURRING": int(persistence_dist.get("RECURRING", 0)),
            "SHORT_LIVED": int(persistence_dist.get("SHORT_LIVED", 0)),
            "INSUFFICIENT_OBSERVATIONS": int(persistence_dist.get("INSUFFICIENT_OBSERVATIONS", 0)),
        },
        "facility_type_counts_among_observed_facilities": {t: int(type_dist.get(t, 0)) for t in FACILITY_TYPES},
        "confidence_composition_among_confirmed_events": {
            "HIGH": int(confidence_dist.get("HIGH", 0)),
            "MEDIUM": int(confidence_dist.get("MEDIUM", 0)),
            "LOW": int(confidence_dist.get("LOW", 0)),
        },
        "temporal_coverage": {
            "first_historical_observation": None if first_obs is None or pd.isna(first_obs) else first_obs.isoformat(),
            "last_historical_observation": None if last_obs is None or pd.isna(last_obs) else last_obs.isoformat(),
        },
        "configuration": {
            "minimum_observations_for_limited_history": config.min_observations_for_limited_history,
            "minimum_observations_for_established_baseline": config.min_observations_for_established_baseline,
            "events_path": str(config.events_path),
            "facilities_path": str(config.facilities_path),
            "rationale": config.describe_rationale(),
        },
        "performance": {"processing_seconds": round(processing_seconds, 3)},
        "reproducibility": {
            "deterministic": True,
            "notes": (
                "Given the same event table, facility table and configuration, "
                "every fingerprint statistic, monthly-profile row and report "
                "value above is identical across repeated runs -- all "
                "aggregation uses vectorized pandas groupby operations, sorted "
                "deterministically by facility_id (and facility_id, month for "
                "the monthly profile), never unordered set/dict iteration."
            ),
        },
        "limitations": [
            "DESCRIPTIVE ONLY, NOT ANOMALY DETECTION: this stage characterizes "
            "each facility's own historical observations; it does not compare "
            "any current/latest event against its facility's fingerprint or "
            "flag anything as unusual. That comparison is GIFT Stage I.4, not "
            "implemented here.",
            "Historical FIRMS observations are satellite thermal detections, "
            "not confirmed facility fires or ground-truth industrial "
            "activity -- a facility's fingerprint describes what was "
            "detected nearby, not what the facility itself did.",
            "The underlying Stage I.2 spatial association is contextual "
            "evidence, not causal proof; a fingerprint's event_count only "
            "counts events for which Stage I.2 confidently selected this "
            "single facility (WITHIN_FACILITY/INTERSECTS_FACILITY/"
            "NEAR_FACILITY) -- AMBIGUOUS and unassociated events are "
            "deliberately excluded from every statistic here, never "
            "attributed to every plausible candidate.",
            "OSM facility coverage is incomplete and of variable geometry "
            "quality (see Stage I.1); a NO_OBSERVATIONS or "
            "INSUFFICIENT_HISTORY facility may simply reflect sparse OSM/"
            "FIRMS coverage, not genuine absence of thermal activity.",
            "PERSISTENT/RECURRING event labels (Stage G.1) describe an "
            "*observed* satellite-detection temporal pattern, not proof "
            "of continuous physical burning.",
            "LIMITED_HISTORY (and especially INSUFFICIENT_HISTORY) "
            "facilities do not have a statistically robust baseline -- "
            "median/MAD/quantile statistics for a handful of events should "
            "be read with that in mind, not as a stable long-run baseline.",
            "No anomaly score, source classification (industrial/wildfire/"
            "agricultural), risk score, or any predictive/ML model is "
            "computed anywhere in this stage.",
            "min_observations_for_limited_history/min_observations_for_"
            "established_baseline are engineering thresholds chosen for "
            "transparency, not scientifically validated minimum sample sizes.",
            "day_event_fraction/night_event_fraction are not guaranteed to "
            "sum to 1 for a facility -- MIXED/UNKNOWN-classified events "
            "remain in the event_count denominator but contribute to "
            "neither numerator; see facility_fingerprint.py for the exact "
            "day/night classification rule.",
            "Each confirmed event is attributed to a single calendar month "
            "(the UTC month of its event_start) for active_month_count and "
            "the monthly profile table -- long-running persistent events "
            "spanning multiple months are not split/duplicated across every "
            "month they touch.",
        ],
    }
    return _to_jsonable(report)


def save_report(report: dict[str, Any], path: str | Path) -> None:
    """Write the report dict to disk as pretty-printed JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=False)
