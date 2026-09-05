"""
Aggregate anomaly scoring, status, and confidence for GIFT Stage I.4.

The anomaly score is a weighted mean of available feature-level robust
deviation indices. Missing features do NOT contribute zero — they are
excluded and the remaining weights are renormalized.

anomaly_score ≠ risk_score. This module produces deviation evidence only.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.anomaly_detection.config import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_NONE,
    HISTORY_ESTABLISHED,
    HISTORY_INSUFFICIENT,
    HISTORY_LIMITED,
    HISTORY_NONE,
    INSUFFICIENT_HISTORY,
    AnomalyConfig,
)
from src.anomaly_detection.temporal_baseline import EventScoreInputs

FEATURE_KEYS: tuple[str, ...] = (
    "peak_frp",
    "event_size",
    "duration",
    "distance",
    "persistence",
    "monthly",
)


@dataclass(frozen=True)
class AnomalyScoreResult:
    anomaly_score: float | None
    anomaly_status: str
    anomaly_confidence: str
    features_evaluated: int
    features_available: int
    feature_names_evaluated: str


def _feature_deviation_map(inputs: EventScoreInputs) -> dict[str, float | None]:
    return {
        "peak_frp": inputs.peak_frp_deviation,
        "event_size": inputs.event_size_deviation,
        "duration": inputs.duration_deviation,
        "distance": inputs.distance_deviation,
        "persistence": inputs.persistence_deviation,
        "monthly": inputs.monthly_deviation,
    }


def compute_anomaly_score(inputs: EventScoreInputs, config: AnomalyConfig) -> AnomalyScoreResult:
    """Aggregate feature deviations into score / status / confidence."""
    history = inputs.baseline_history_status
    if history in (HISTORY_NONE, HISTORY_INSUFFICIENT):
        return AnomalyScoreResult(
            anomaly_score=None,
            anomaly_status=INSUFFICIENT_HISTORY,
            anomaly_confidence=CONFIDENCE_NONE,
            features_evaluated=0,
            features_available=0,
            feature_names_evaluated="",
        )

    deviations = _feature_deviation_map(inputs)
    weighted_sum = 0.0
    weight_total = 0.0
    evaluated: list[str] = []

    for name in FEATURE_KEYS:
        value = deviations[name]
        weight = config.feature_weights.get(name, 0.0)
        if value is None or weight <= 0:
            continue
        weighted_sum += float(value) * weight
        weight_total += weight
        evaluated.append(name)

    if weight_total <= 0 or not evaluated:
        return AnomalyScoreResult(
            anomaly_score=None,
            anomaly_status=INSUFFICIENT_HISTORY,
            anomaly_confidence=CONFIDENCE_NONE,
            features_evaluated=0,
            features_available=inputs.features_available,
            feature_names_evaluated="",
        )

    score = weighted_sum / weight_total
    # Guard against tiny negative floats from numeric noise; scores are abs-based.
    if score < 0:
        score = 0.0

    status = config.classify_anomaly_status(score, history)
    confidence = _classify_confidence(history, len(evaluated), config)
    return AnomalyScoreResult(
        anomaly_score=float(score),
        anomaly_status=status,
        anomaly_confidence=confidence,
        features_evaluated=len(evaluated),
        features_available=inputs.features_available,
        feature_names_evaluated=",".join(evaluated),
    )


def _classify_confidence(history_status: str, features_evaluated: int, config: AnomalyConfig) -> str:
    """Evidence-quality confidence — not a fire probability."""
    if history_status in (HISTORY_NONE, HISTORY_INSUFFICIENT) or features_evaluated == 0:
        return CONFIDENCE_NONE
    if history_status == HISTORY_LIMITED:
        return CONFIDENCE_LOW if features_evaluated < 3 else CONFIDENCE_MEDIUM
    # ESTABLISHED_BASELINE
    if features_evaluated >= 4:
        return CONFIDENCE_HIGH
    if features_evaluated >= 2:
        return CONFIDENCE_MEDIUM
    return CONFIDENCE_LOW


def unavailable_result(reason: str) -> AnomalyScoreResult:
    """Score result for events that cannot receive facility-specific scoring."""
    return AnomalyScoreResult(
        anomaly_score=None,
        anomaly_status=INSUFFICIENT_HISTORY,
        anomaly_confidence=CONFIDENCE_NONE,
        features_evaluated=0,
        features_available=0,
        feature_names_evaluated="",
    )
