"""Tests for Stage I.4 anomaly config, scoring, status, and confidence."""

from __future__ import annotations

import pytest

from src.anomaly_detection.anomaly_scoring import compute_anomaly_score
from src.anomaly_detection.config import (
    ANOMALOUS,
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_NONE,
    ELEVATED,
    HISTORY_ESTABLISHED,
    HISTORY_INSUFFICIENT,
    HISTORY_LIMITED,
    INSUFFICIENT_HISTORY,
    NORMAL,
    AnomalyConfig,
)
from src.anomaly_detection.temporal_baseline import EventScoreInputs


def _inputs(**kwargs) -> EventScoreInputs:
    base = dict(
        event_id="E1",
        facility_id="F1",
        baseline_observation_count=10,
        baseline_history_status=HISTORY_ESTABLISHED,
        anomaly_unavailable_reason=None,
        peak_frp_deviation=1.0,
        event_size_deviation=1.0,
        duration_deviation=1.0,
        distance_deviation=1.0,
        persistence_deviation=0.0,
        monthly_deviation=None,
        features_available=5,
    )
    base.update(kwargs)
    return EventScoreInputs(**base)


def test_config_defaults() -> None:
    config = AnomalyConfig()
    assert config.normal_max_score == 2.0
    assert config.elevated_max_score == 3.5
    assert abs(sum(config.feature_weights.values()) - 1.0) < 1e-9


def test_config_invalid_thresholds_raise() -> None:
    with pytest.raises(ValueError):
        AnomalyConfig(normal_max_score=5.0, elevated_max_score=3.0)
    with pytest.raises(ValueError):
        AnomalyConfig(min_observations_for_limited_history=10, min_observations_for_established_baseline=5)


@pytest.mark.parametrize(
    ("prior", "expected"),
    [
        (0, "NO_PRIOR_OBSERVATIONS"),
        (1, "INSUFFICIENT_HISTORY"),
        (2, "INSUFFICIENT_HISTORY"),
        (3, "LIMITED_HISTORY"),
        (9, "LIMITED_HISTORY"),
        (10, "ESTABLISHED_BASELINE"),
    ],
)
def test_history_status_boundaries(prior: int, expected: str) -> None:
    assert AnomalyConfig().classify_history_status(prior) == expected


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, NORMAL),
        (1.99, NORMAL),
        (2.0, ELEVATED),
        (3.49, ELEVATED),
        (3.5, ANOMALOUS),
        (10.0, ANOMALOUS),
    ],
)
def test_anomaly_status_boundaries(score: float, expected: str) -> None:
    assert AnomalyConfig().classify_anomaly_status(score, HISTORY_ESTABLISHED) == expected


def test_insufficient_history_yields_none_score() -> None:
    config = AnomalyConfig()
    result = compute_anomaly_score(
        _inputs(baseline_history_status=HISTORY_INSUFFICIENT, baseline_observation_count=1),
        config,
    )
    assert result.anomaly_score is None
    assert result.anomaly_status == INSUFFICIENT_HISTORY
    assert result.anomaly_confidence == CONFIDENCE_NONE


def test_missing_features_do_not_contribute_zero() -> None:
    config = AnomalyConfig()
    # Only peak_frp available with high deviation; others None.
    result = compute_anomaly_score(
        _inputs(
            peak_frp_deviation=4.0,
            event_size_deviation=None,
            duration_deviation=None,
            distance_deviation=None,
            persistence_deviation=None,
            monthly_deviation=None,
            features_available=1,
        ),
        config,
    )
    assert result.anomaly_score == pytest.approx(4.0)
    assert result.features_evaluated == 1
    assert result.feature_names_evaluated == "peak_frp"


def test_weighted_mean_of_available_features() -> None:
    config = AnomalyConfig()
    # peak_frp weight 0.30, event_size 0.20 → renormalize
    result = compute_anomaly_score(
        _inputs(
            peak_frp_deviation=2.0,
            event_size_deviation=4.0,
            duration_deviation=None,
            distance_deviation=None,
            persistence_deviation=None,
            monthly_deviation=None,
        ),
        config,
    )
    expected = (2.0 * 0.30 + 4.0 * 0.20) / (0.30 + 0.20)
    assert result.anomaly_score == pytest.approx(expected)


def test_confidence_established_high_when_many_features() -> None:
    result = compute_anomaly_score(_inputs(), AnomalyConfig())
    assert result.anomaly_confidence in (CONFIDENCE_MEDIUM, CONFIDENCE_HIGH)
    # 5 features available with weights → should be HIGH (>=4)
    assert result.anomaly_confidence == CONFIDENCE_HIGH


def test_confidence_limited_is_low_or_medium() -> None:
    result = compute_anomaly_score(
        _inputs(baseline_history_status=HISTORY_LIMITED, baseline_observation_count=5),
        AnomalyConfig(),
    )
    assert result.anomaly_confidence in (CONFIDENCE_LOW, CONFIDENCE_MEDIUM)
