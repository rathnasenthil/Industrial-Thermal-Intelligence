"""
Walk-forward / prior-only temporal baseline for GIFT Stage I.4.

CRITICAL TEMPORAL RULE
------------------------------------------------------------------------
For each facility, confirmed associated events are sorted by
(event_start ASC, event_id ASC). Event N is scored using ONLY events
1..N-1 at that facility. The current event is NEVER in its own baseline.

Correct order:
  1. obtain current event
  2. retrieve previous facility history
  3. calculate baseline
  4. score current event
  5. record result
  6. THEN add current event to historical state

AMBIGUOUS and NO_FACILITY_ASSOCIATION events never enter any facility's
confirmed history and never receive a facility-specific score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.anomaly_detection.config import (
    AnomalyConfig,
    CONFIRMED_ASSOCIATION_METHODS,
    HISTORY_INSUFFICIENT,
    HISTORY_NONE,
    REASON_INSUFFICIENT_PRIOR,
)
from src.anomaly_detection.robust_deviation import (
    FeatureDeviation,
    compute_baseline_stats,
    persistence_rarity_deviation,
    robust_deviation,
)

# Stage G / I.2 column names used for walk-forward features.
COL_EVENT_ID = "event_id"
COL_START = "event_start"
COL_PEAK_FRP = "peak_frp"
COL_SIZE = "detection_count"
COL_DURATION = "observed_duration_hours"
COL_DISTANCE = "facility_distance_km"
COL_PERSISTENCE = "persistence_label"
COL_FACILITY = "facility_id"
COL_METHOD = "facility_association_method"


@dataclass
class EventScoreInputs:
    """Per-event walk-forward feature deviations + baseline metadata."""

    event_id: str
    facility_id: str | None
    baseline_observation_count: int
    baseline_history_status: str
    anomaly_unavailable_reason: str | None

    peak_frp_deviation: float | None = None
    event_size_deviation: float | None = None
    duration_deviation: float | None = None
    distance_deviation: float | None = None
    persistence_deviation: float | None = None
    monthly_deviation: float | None = None

    peak_frp_method: str | None = None
    event_size_method: str | None = None
    duration_method: str | None = None
    distance_method: str | None = None
    persistence_method: str | None = None
    monthly_method: str | None = None

    baseline_peak_frp_median: float | None = None
    baseline_peak_frp_mad: float | None = None
    baseline_event_size_median: float | None = None
    baseline_event_size_mad: float | None = None
    baseline_duration_median: float | None = None
    baseline_duration_mad: float | None = None
    baseline_distance_median: float | None = None
    baseline_distance_mad: float | None = None

    features_available: int = 0
    feature_names_available: str = ""


def is_confirmed_association(row: pd.Series) -> bool:
    """True iff Stage I.2 selected exactly one facility for this event."""
    facility_id = row.get(COL_FACILITY)
    method = row.get(COL_METHOD)
    if facility_id is None or (isinstance(facility_id, float) and np.isnan(facility_id)):
        return False
    if pd.isna(facility_id) or str(facility_id).strip() == "" or str(facility_id) == "nan":
        return False
    return str(method) in CONFIRMED_ASSOCIATION_METHODS


def _as_float(value: object) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if np.isnan(f):
        return None
    return f


def _apply_feature(
    name: str,
    current: float | None,
    prior_values: list[float],
    config: AnomalyConfig,
) -> tuple[FeatureDeviation, BaselineStats | None]:
    baseline = compute_baseline_stats(prior_values)
    if baseline.count < config.min_observations_for_limited_history:
        # Not enough prior numeric observations for this feature.
        return FeatureDeviation(None, None, baseline if baseline.count else None), baseline
    return robust_deviation(current, baseline, config), baseline


def score_facility_events_walk_forward(
    facility_events: pd.DataFrame,
    config: AnomalyConfig,
) -> list[EventScoreInputs]:
    """Score all confirmed events for one facility in chronological order.

    ``facility_events`` must contain only confirmed associations for a
    single facility_id. Results are returned in the same chronological
    order used for scoring (not the original CSV row order).
    """
    if facility_events.empty:
        return []

    ordered = facility_events.sort_values(
        [COL_START, COL_EVENT_ID], kind="mergesort"
    ).reset_index(drop=True)

    prior_frp: list[float] = []
    prior_size: list[float] = []
    prior_duration: list[float] = []
    prior_distance: list[float] = []
    prior_labels: list[str] = []
    # month -> list of prior peak_frp values (prior-only monthly baseline)
    prior_monthly_frp: dict[int, list[float]] = {}

    results: list[EventScoreInputs] = []

    for _, row in ordered.iterrows():
        event_id = str(row[COL_EVENT_ID])
        facility_id = str(row[COL_FACILITY])
        prior_count = len(prior_labels)
        history_status = config.classify_history_status(prior_count)

        result = EventScoreInputs(
            event_id=event_id,
            facility_id=facility_id,
            baseline_observation_count=prior_count,
            baseline_history_status=history_status,
            anomaly_unavailable_reason=None,
        )

        if history_status in (HISTORY_NONE, HISTORY_INSUFFICIENT):
            result.anomaly_unavailable_reason = REASON_INSUFFICIENT_PRIOR
            results.append(result)
            # THEN add current event to history (after scoring / skipping).
            _append_prior(
                row,
                prior_frp,
                prior_size,
                prior_duration,
                prior_distance,
                prior_labels,
                prior_monthly_frp,
            )
            continue

        # --- Numeric feature deviations (overall prior baseline) ---
        frp_dev, frp_base = _apply_feature(
            "peak_frp", _as_float(row.get(COL_PEAK_FRP)), prior_frp, config
        )
        size_dev, size_base = _apply_feature(
            "event_size", _as_float(row.get(COL_SIZE)), prior_size, config
        )
        dur_dev, dur_base = _apply_feature(
            "duration", _as_float(row.get(COL_DURATION)), prior_duration, config
        )
        dist_dev, dist_base = _apply_feature(
            "distance", _as_float(row.get(COL_DISTANCE)), prior_distance, config
        )

        result.peak_frp_deviation = frp_dev.deviation
        result.peak_frp_method = frp_dev.method
        result.event_size_deviation = size_dev.deviation
        result.event_size_method = size_dev.method
        result.duration_deviation = dur_dev.deviation
        result.duration_method = dur_dev.method
        result.distance_deviation = dist_dev.deviation
        result.distance_method = dist_dev.method

        if frp_base is not None and frp_base.count > 0:
            result.baseline_peak_frp_median = frp_base.median
            result.baseline_peak_frp_mad = frp_base.mad
        if size_base is not None and size_base.count > 0:
            result.baseline_event_size_median = size_base.median
            result.baseline_event_size_mad = size_base.mad
        if dur_base is not None and dur_base.count > 0:
            result.baseline_duration_median = dur_base.median
            result.baseline_duration_mad = dur_base.mad
        if dist_base is not None and dist_base.count > 0:
            result.baseline_distance_median = dist_base.median
            result.baseline_distance_mad = dist_base.mad

        # --- Persistence rarity (consumes G.1 labels; does not recompute) ---
        pers_dev = persistence_rarity_deviation(
            row.get(COL_PERSISTENCE),
            prior_labels,
            min_prior=config.min_observations_for_limited_history,
        )
        result.persistence_deviation = pers_dev.deviation
        result.persistence_method = pers_dev.method

        # --- Monthly prior-only peak_frp deviation ---
        # Uses only earlier events in the same calendar month at this facility.
        # Future same-month observations never enter this baseline.
        start_dt = pd.to_datetime(row[COL_START], utc=True)
        month = int(start_dt.month)
        month_priors = prior_monthly_frp.get(month, [])
        if len(month_priors) >= config.min_monthly_prior_observations:
            month_baseline = compute_baseline_stats(month_priors)
            month_dev = robust_deviation(_as_float(row.get(COL_PEAK_FRP)), month_baseline, config)
            result.monthly_deviation = month_dev.deviation
            result.monthly_method = month_dev.method
        else:
            result.monthly_deviation = None
            result.monthly_method = None

        available = []
        for fname, val in (
            ("peak_frp", result.peak_frp_deviation),
            ("event_size", result.event_size_deviation),
            ("duration", result.duration_deviation),
            ("distance", result.distance_deviation),
            ("persistence", result.persistence_deviation),
            ("monthly", result.monthly_deviation),
        ):
            if val is not None:
                available.append(fname)
        result.features_available = len(available)
        result.feature_names_available = ",".join(available)

        results.append(result)

        # THEN add current event to historical state.
        _append_prior(
            row,
            prior_frp,
            prior_size,
            prior_duration,
            prior_distance,
            prior_labels,
            prior_monthly_frp,
        )

    return results


def _append_prior(
    row: pd.Series,
    prior_frp: list[float],
    prior_size: list[float],
    prior_duration: list[float],
    prior_distance: list[float],
    prior_labels: list[str],
    prior_monthly_frp: dict[int, list[float]],
) -> None:
    """Append the just-scored event to prior state (never before scoring)."""
    frp = _as_float(row.get(COL_PEAK_FRP))
    size = _as_float(row.get(COL_SIZE))
    dur = _as_float(row.get(COL_DURATION))
    dist = _as_float(row.get(COL_DISTANCE))
    if frp is not None:
        prior_frp.append(frp)
    if size is not None:
        prior_size.append(size)
    if dur is not None:
        prior_duration.append(dur)
    if dist is not None:
        prior_distance.append(dist)
    label = row.get(COL_PERSISTENCE)
    if label is not None and not (isinstance(label, float) and np.isnan(label)):
        prior_labels.append(str(label))
    start_dt = pd.to_datetime(row[COL_START], utc=True)
    month = int(start_dt.month)
    if frp is not None:
        prior_monthly_frp.setdefault(month, []).append(frp)


def walk_forward_score_all_facilities(
    confirmed_events: pd.DataFrame,
    config: AnomalyConfig,
) -> dict[str, EventScoreInputs]:
    """Score every confirmed associated event; return map event_id → inputs.

    Groups by facility_id and runs walk-forward independently per facility.
    Ambiguous / unassociated events must NOT be present in ``confirmed_events``.
    """
    out: dict[str, EventScoreInputs] = {}
    if confirmed_events.empty:
        return out

    # Deterministic facility iteration order.
    for facility_id, group in confirmed_events.groupby(COL_FACILITY, sort=True):
        for scored in score_facility_events_walk_forward(group, config):
            out[scored.event_id] = scored
    return out
