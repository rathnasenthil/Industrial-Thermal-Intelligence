"""
Deterministic human-readable anomaly explanations for GIFT Stage I.4.

Explanations are generated from feature contributions with fixed templates.
No LLM is used.
"""

from __future__ import annotations

from src.anomaly_detection.anomaly_scoring import AnomalyScoreResult
from src.anomaly_detection.config import (
    ANOMALOUS,
    ELEVATED,
    INSUFFICIENT_HISTORY,
    NORMAL,
    REASON_AMBIGUOUS,
    REASON_INSUFFICIENT_PRIOR,
    REASON_NO_FACILITY,
)
from src.anomaly_detection.temporal_baseline import EventScoreInputs

_FEATURE_LABELS: dict[str, str] = {
    "peak_frp": "peak FRP",
    "event_size": "event detection count",
    "duration": "event duration",
    "distance": "event-to-facility distance",
    "persistence": "persistence behaviour",
    "monthly": "same-month historical peak FRP",
}


def build_explanation(
    inputs: EventScoreInputs,
    score_result: AnomalyScoreResult,
    unavailable_reason: str | None = None,
) -> str:
    """Return a deterministic explanation string for one event."""
    reason = unavailable_reason or inputs.anomaly_unavailable_reason
    if score_result.anomaly_status == INSUFFICIENT_HISTORY:
        if reason == REASON_NO_FACILITY:
            return (
                "Facility-specific anomaly scoring unavailable: no confirmed "
                "facility association (NO_FACILITY_ASSOCIATION). Absence of an "
                "OSM facility match is not evidence that the event is non-industrial."
            )
        if reason == REASON_AMBIGUOUS:
            return (
                "Facility-specific anomaly scoring unavailable: Stage I.2 marked "
                "the association AMBIGUOUS, so no single facility baseline is used. "
                "Ambiguous candidates are never assigned to a facility history."
            )
        if reason == REASON_INSUFFICIENT_PRIOR or inputs.baseline_observation_count < 3:
            return (
                f"Insufficient facility history for reliable anomaly assessment "
                f"({inputs.baseline_observation_count} prior confirmed observation(s) "
                f"at this facility; engineering minimum for scoring is 3)."
            )
        return "Insufficient evidence to compute a facility-specific anomaly score."

    deviations = {
        "peak_frp": inputs.peak_frp_deviation,
        "event_size": inputs.event_size_deviation,
        "duration": inputs.duration_deviation,
        "distance": inputs.distance_deviation,
        "persistence": inputs.persistence_deviation,
        "monthly": inputs.monthly_deviation,
    }
    contributors = [
        (name, deviations[name])
        for name in score_result.feature_names_evaluated.split(",")
        if name and deviations.get(name) is not None
    ]
    contributors.sort(key=lambda x: float(x[1]), reverse=True)  # type: ignore[arg-type]

    status = score_result.anomaly_status
    score = score_result.anomaly_score
    score_txt = f"{score:.2f}" if score is not None else "n/a"

    if status == NORMAL:
        lead = (
            f"Normal relative to available historical facility behaviour "
            f"(anomaly_score={score_txt} on a robust deviation index)."
        )
    elif status == ELEVATED:
        lead = (
            f"Elevated relative to prior facility behaviour "
            f"(anomaly_score={score_txt} on a robust deviation index)."
        )
    elif status == ANOMALOUS:
        lead = (
            f"Anomalous relative to prior facility behaviour "
            f"(anomaly_score={score_txt} on a robust deviation index)."
        )
    else:
        lead = f"Status={status} (anomaly_score={score_txt})."

    if not contributors:
        return lead + " No individual feature deviations were available."

    parts: list[str] = []
    for name, value in contributors[:3]:
        label = _FEATURE_LABELS.get(name, name)
        parts.append(f"{label} deviation={float(value):.2f}")

    detail = " Top feature contributions: " + "; ".join(parts) + "."
    if status == NORMAL:
        caveat = (
            " Deviations are within the engineering NORMAL band relative to "
            "prior confirmed associations at this facility; this does not "
            "classify the source as an industrial fire (or as safe)."
        )
    else:
        caveat = (
            " This indicates unusual thermal behaviour relative to the facility's "
            "prior confirmed associations; it does not classify the source as an "
            "industrial fire."
        )
    return lead + detail + caveat
