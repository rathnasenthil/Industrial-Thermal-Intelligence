"""
Incremental Stage I.4 temporal anomaly detection (AIML realtime adapter).

Walk-forward prior-only scoring for **one** event at a facility.

I.4 is temporal deviation evidence — not industrial-fire classification,
not risk scoring, and not I.3 fingerprinting.

Why realtime I.4 cannot replay the full batch pipeline
------------------------------------------------------
Batch ``run_anomaly_detection()`` scores every confirmed event across the
Stage I.2 table and rebuilds a full report. On each NRT poll only the
affected event needs scoring. Re-running the batch entrypoint would
rewrite historical I.4 fields for unrelated events.

This adapter calls ``score_facility_events_walk_forward`` on the facility's
confirmed events (including the current row), then returns only the
current event's scored inputs + aggregate score/explanation.

The current event is never in its own baseline — that rule is enforced
inside the batch walk-forward scorer (score, then append prior).

I.3 fingerprint tables are never read here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import pandas as pd

from src.anomaly_detection.anomaly_explanation import build_explanation
from src.anomaly_detection.anomaly_scoring import (
    AnomalyScoreResult,
    compute_anomaly_score,
    unavailable_result,
)
from src.anomaly_detection.config import (
    DEFAULT_CONFIG,
    REASON_AMBIGUOUS,
    REASON_NO_FACILITY,
    AnomalyConfig,
)
from src.anomaly_detection.temporal_baseline import (
    EventScoreInputs,
    is_confirmed_association,
    score_facility_events_walk_forward,
)

REQUIRED_COLUMNS: tuple[str, ...] = (
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


@dataclass(frozen=True)
class AnomalyResult:
    """I.4 fields for one event (realtime ThermalEvent schema subset)."""

    event_id: str
    baseline_observation_count: int
    baseline_history_status: str
    anomaly_unavailable_reason: str | None
    anomaly_score: float | None
    anomaly_status: str
    anomaly_confidence: str
    peak_frp_deviation: float | None
    event_size_deviation: float | None
    duration_deviation: float | None
    distance_deviation: float | None
    persistence_deviation: float | None
    monthly_deviation: float | None
    features_available: int
    features_evaluated: int
    anomaly_explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def unavailable_anomaly_result(
    event_id: str,
    *,
    reason: str,
) -> AnomalyResult:
    """Build the batch-equivalent unavailable I.4 payload."""
    score = unavailable_result(reason)
    if reason == REASON_AMBIGUOUS:
        explanation = _EXPL_AMBIGUOUS
    elif reason == REASON_NO_FACILITY:
        explanation = _EXPL_NO_FACILITY
    else:
        explanation = "Facility-specific anomaly scoring unavailable."
    return AnomalyResult(
        event_id=str(event_id),
        baseline_observation_count=0,
        baseline_history_status="NOT_APPLICABLE",
        anomaly_unavailable_reason=reason,
        anomaly_score=None,
        anomaly_status=score.anomaly_status,
        anomaly_confidence=score.anomaly_confidence,
        peak_frp_deviation=None,
        event_size_deviation=None,
        duration_deviation=None,
        distance_deviation=None,
        persistence_deviation=None,
        monthly_deviation=None,
        features_available=0,
        features_evaluated=0,
        anomaly_explanation=explanation,
    )


def _from_scored(
    inputs: EventScoreInputs,
    score: AnomalyScoreResult,
    explanation: str,
) -> AnomalyResult:
    return AnomalyResult(
        event_id=inputs.event_id,
        baseline_observation_count=int(inputs.baseline_observation_count),
        baseline_history_status=str(inputs.baseline_history_status),
        anomaly_unavailable_reason=inputs.anomaly_unavailable_reason,
        anomaly_score=score.anomaly_score,
        anomaly_status=score.anomaly_status,
        anomaly_confidence=score.anomaly_confidence,
        peak_frp_deviation=inputs.peak_frp_deviation,
        event_size_deviation=inputs.event_size_deviation,
        duration_deviation=inputs.duration_deviation,
        distance_deviation=inputs.distance_deviation,
        persistence_deviation=inputs.persistence_deviation,
        monthly_deviation=inputs.monthly_deviation,
        features_available=int(
            score.features_available if score.features_available else inputs.features_available
        ),
        features_evaluated=int(score.features_evaluated),
        anomaly_explanation=explanation,
    )


def process_event_anomaly(
    events_df: pd.DataFrame,
    event_id: str,
    *,
    config: Optional[AnomalyConfig] = None,
) -> AnomalyResult:
    """
    Score **one** confirmed-facility event with batch I.4 walk-forward semantics.

    ``events_df`` must contain confirmed associations for a single facility
    and **include** the current ``event_id`` row. Walk-forward sorts by
    ``(event_start, event_id)`` and scores the current event against only
    earlier rows — the current event never enters its own baseline.

    For AMBIGUOUS / NO_FACILITY_ASSOCIATION, callers should use
    ``unavailable_anomaly_result`` instead of this function.
    """
    cfg = config or DEFAULT_CONFIG
    eid = str(event_id)

    if events_df is None or events_df.empty:
        raise ValueError("events_df must contain the current confirmed event.")

    for col in REQUIRED_COLUMNS:
        if col not in events_df.columns:
            raise ValueError(f"Events dataframe missing required column: {col}")

    work = events_df.copy()
    confirmed_mask = work.apply(is_confirmed_association, axis=1)
    confirmed = work.loc[confirmed_mask].copy()
    if confirmed.empty or not (confirmed["event_id"].astype(str) == eid).any():
        raise ValueError(
            f"event_id={eid} is not present as a confirmed association in events_df."
        )

    # Guard: only one facility in the frame (realtime contract).
    facility_ids = confirmed["facility_id"].astype(str).unique()
    if len(facility_ids) != 1:
        raise ValueError(
            f"Realtime I.4 expects exactly one facility; got {list(facility_ids)}."
        )

    scored = score_facility_events_walk_forward(confirmed, cfg)
    inputs = next((s for s in scored if s.event_id == eid), None)
    if inputs is None:
        raise RuntimeError(f"Walk-forward did not produce a result for event_id={eid}")

    score = compute_anomaly_score(inputs, cfg)
    explanation = build_explanation(inputs, score)
    return _from_scored(inputs, score, explanation)
