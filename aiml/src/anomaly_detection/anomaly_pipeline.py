"""
End-to-end Temporal Deviation & Anomaly Detection pipeline (GIFT Stage I.4).

Walk-forward prior-only scoring over Stage I.2 confirmed associations.
Does not modify Stage G / G.1 / I.1 / I.2 / I.3 outputs or logic.
Does not perform source classification, risk scoring, or ML.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.anomaly_detection.anomaly_explanation import build_explanation
from src.anomaly_detection.anomaly_report import build_anomaly_report
from src.anomaly_detection.anomaly_scoring import compute_anomaly_score
from src.anomaly_detection.config import (
    AnomalyConfig,
    CONFIRMED_ASSOCIATION_METHODS,
    REASON_AMBIGUOUS,
    REASON_NO_FACILITY,
)
from src.anomaly_detection.temporal_baseline import (
    EventScoreInputs,
    walk_forward_score_all_facilities,
)

REQUIRED_EVENT_COLUMNS: tuple[str, ...] = (
    "event_id",
    "event_start",
    "event_end",
    "peak_frp",
    "detection_count",
    "observed_duration_hours",
    "persistence_label",
    "facility_id",
    "facility_association_method",
    "facility_distance_km",
)

I4_APPEND_COLUMNS: tuple[str, ...] = (
    "baseline_observation_count",
    "baseline_history_status",
    "anomaly_unavailable_reason",
    "anomaly_score",
    "anomaly_status",
    "anomaly_confidence",
    "peak_frp_deviation",
    "event_size_deviation",
    "duration_deviation",
    "distance_deviation",
    "persistence_deviation",
    "monthly_deviation",
    "features_available",
    "features_evaluated",
    "feature_names_evaluated",
    "baseline_peak_frp_median",
    "baseline_peak_frp_mad",
    "baseline_event_size_median",
    "baseline_event_size_mad",
    "baseline_duration_median",
    "baseline_duration_mad",
    "baseline_distance_median",
    "baseline_distance_mad",
    "anomaly_explanation",
)

_NUMERIC_I4_COLUMNS: tuple[str, ...] = (
    "anomaly_score",
    "peak_frp_deviation",
    "event_size_deviation",
    "duration_deviation",
    "distance_deviation",
    "persistence_deviation",
    "monthly_deviation",
    "baseline_peak_frp_median",
    "baseline_peak_frp_mad",
    "baseline_event_size_median",
    "baseline_event_size_mad",
    "baseline_duration_median",
    "baseline_duration_mad",
    "baseline_distance_median",
    "baseline_distance_mad",
    "baseline_observation_count",
    "features_available",
    "features_evaluated",
)

_EXPL_AMBIGUOUS = (
    "Facility-specific anomaly scoring unavailable: Stage I.2 marked "
    "the association AMBIGUOUS, so no single facility baseline is used. "
    "Ambiguous candidates are never assigned to a facility history."
)
_EXPL_NO_FACILITY = (
    "Facility-specific anomaly scoring unavailable: no confirmed "
    "facility association (NO_FACILITY_ASSOCIATION). Absence of an "
    "OSM facility match is not evidence that the event is non-industrial."
)


@dataclass
class AnomalyResult:
    events_df: pd.DataFrame
    report: dict[str, Any]


def load_events(path: str | Path) -> pd.DataFrame:
    events_path = Path(path)
    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")
    df = pd.read_csv(events_path)
    missing = [c for c in REQUIRED_EVENT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"'{events_path}' is missing required column(s): {missing}. "
            "Stage I.4 expects thermal_events_with_facility_association.csv."
        )
    return df


def load_fingerprints(path: str | Path) -> pd.DataFrame:
    """Load I.3 fingerprints for provenance/reporting only (not scoring baseline)."""
    fp_path = Path(path)
    if not fp_path.exists():
        raise FileNotFoundError(f"Facility fingerprints file not found: {fp_path}")
    return pd.read_csv(fp_path)


def _confirmed_mask(df: pd.DataFrame) -> pd.Series:
    facility_ok = df["facility_id"].notna() & (df["facility_id"].astype(str).str.strip() != "")
    facility_ok &= df["facility_id"].astype(str) != "nan"
    method_ok = df["facility_association_method"].isin(CONFIRMED_ASSOCIATION_METHODS)
    return facility_ok & method_ok


def run_anomaly_detection(
    events_df: pd.DataFrame,
    config: AnomalyConfig,
    *,
    fingerprints_df: pd.DataFrame | None = None,
    events_input_path: str = "<in-memory>",
    fingerprints_input_path: str = "<in-memory>",
) -> AnomalyResult:
    """Run Stage I.4 over an already-loaded Stage I.2 events table.

    Preserves every input row and appends I.4 columns. Does not mutate
    ``events_df``. I.3 fingerprints are optional metadata for the report
    and are never used as the walk-forward scoring baseline.
    """
    start = time.perf_counter()
    working = events_df.copy()
    event_ids = working["event_id"].astype(str)

    confirmed_mask = _confirmed_mask(working)
    confirmed = working.loc[confirmed_mask]
    scored_map = walk_forward_score_all_facilities(confirmed, config)

    scored_rows: list[dict[str, Any]] = []
    for event_id, inputs in scored_map.items():
        score_result = compute_anomaly_score(inputs, config)
        explanation = build_explanation(inputs, score_result)
        scored_rows.append(_row_from_inputs(inputs, score_result, explanation))

    if scored_rows:
        scored_df = pd.DataFrame(scored_rows)
        for col in _NUMERIC_I4_COLUMNS:
            if col in scored_df.columns:
                scored_df[col] = pd.to_numeric(scored_df[col], errors="coerce")
    else:
        scored_df = pd.DataFrame(columns=["event_id", *I4_APPEND_COLUMNS])

    # Build per-row unavailable defaults for every event, then left-join scores.
    amb_mask = working["facility_association_method"].astype(str) == "AMBIGUOUS"
    no_mask = ~confirmed_mask & ~amb_mask

    base = pd.DataFrame({"event_id": event_ids.to_numpy()})
    base["baseline_observation_count"] = 0
    base["baseline_history_status"] = "NOT_APPLICABLE"
    base["anomaly_unavailable_reason"] = np.select(
        [amb_mask.to_numpy(), no_mask.to_numpy()],
        [REASON_AMBIGUOUS, REASON_NO_FACILITY],
        default=None,
    )
    base["anomaly_score"] = np.nan
    base["anomaly_status"] = "INSUFFICIENT_HISTORY"
    base["anomaly_confidence"] = "NONE"
    for col in (
        "peak_frp_deviation",
        "event_size_deviation",
        "duration_deviation",
        "distance_deviation",
        "persistence_deviation",
        "monthly_deviation",
        "baseline_peak_frp_median",
        "baseline_peak_frp_mad",
        "baseline_event_size_median",
        "baseline_event_size_mad",
        "baseline_duration_median",
        "baseline_duration_mad",
        "baseline_distance_median",
        "baseline_distance_mad",
    ):
        base[col] = np.nan
    base["features_available"] = 0
    base["features_evaluated"] = 0
    base["feature_names_evaluated"] = ""
    base["anomaly_explanation"] = np.select(
        [amb_mask.to_numpy(), no_mask.to_numpy()],
        [_EXPL_AMBIGUOUS, _EXPL_NO_FACILITY],
        default="",
    )

    if not scored_df.empty:
        # Scored rows replace defaults for matching event_ids.
        scored_indexed = scored_df.set_index("event_id")
        base_indexed = base.set_index("event_id")
        base_indexed.update(scored_indexed)
        # update() leaves anomaly_unavailable_reason as prior default for scored
        # rows that had None — clear it for any event present in scored_df.
        base_indexed.loc[scored_indexed.index, "anomaly_unavailable_reason"] = scored_indexed[
            "anomaly_unavailable_reason"
        ]
        append = base_indexed.reset_index()
    else:
        append = base

    # Restore original working row order before sorting for output.
    append = append.set_index("event_id").reindex(event_ids).reset_index()

    for col in I4_APPEND_COLUMNS:
        working[col] = append[col].to_numpy()

    working = working.sort_values("event_id", kind="mergesort").reset_index(drop=True)

    assert len(working) == len(events_df)
    assert working["event_id"].is_unique
    scored_scores = working["anomaly_score"].dropna()
    assert scored_scores.empty or (scored_scores >= 0).all()

    processing_seconds = time.perf_counter() - start
    report = build_anomaly_report(
        config=config,
        events_input_path=events_input_path,
        fingerprints_input_path=fingerprints_input_path,
        output_df=working,
        processing_seconds=processing_seconds,
    )
    if fingerprints_df is not None:
        report["input"]["fingerprint_facility_count"] = int(fingerprints_df["facility_id"].nunique())

    return AnomalyResult(events_df=working, report=report)


def _row_from_inputs(inputs: EventScoreInputs, score_result, explanation: str) -> dict[str, Any]:
    return {
        "event_id": inputs.event_id,
        "baseline_observation_count": inputs.baseline_observation_count,
        "baseline_history_status": inputs.baseline_history_status,
        "anomaly_unavailable_reason": inputs.anomaly_unavailable_reason,
        "anomaly_score": score_result.anomaly_score,
        "anomaly_status": score_result.anomaly_status,
        "anomaly_confidence": score_result.anomaly_confidence,
        "peak_frp_deviation": inputs.peak_frp_deviation,
        "event_size_deviation": inputs.event_size_deviation,
        "duration_deviation": inputs.duration_deviation,
        "distance_deviation": inputs.distance_deviation,
        "persistence_deviation": inputs.persistence_deviation,
        "monthly_deviation": inputs.monthly_deviation,
        "features_available": (
            score_result.features_available if score_result.features_available else inputs.features_available
        ),
        "features_evaluated": score_result.features_evaluated,
        "feature_names_evaluated": score_result.feature_names_evaluated,
        "baseline_peak_frp_median": inputs.baseline_peak_frp_median,
        "baseline_peak_frp_mad": inputs.baseline_peak_frp_mad,
        "baseline_event_size_median": inputs.baseline_event_size_median,
        "baseline_event_size_mad": inputs.baseline_event_size_mad,
        "baseline_duration_median": inputs.baseline_duration_median,
        "baseline_duration_mad": inputs.baseline_duration_mad,
        "baseline_distance_median": inputs.baseline_distance_median,
        "baseline_distance_mad": inputs.baseline_distance_mad,
        "anomaly_explanation": explanation,
    }


def save_outputs(result: AnomalyResult, events_output_path: str | Path) -> None:
    path = Path(events_output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.events_df.to_csv(path, index=False)
