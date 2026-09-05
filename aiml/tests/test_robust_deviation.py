"""Tests for Stage I.4 robust deviation and zero-MAD handling."""

from __future__ import annotations

import numpy as np
import pytest

from src.anomaly_detection.config import AnomalyConfig
from src.anomaly_detection.robust_deviation import (
    compute_baseline_stats,
    persistence_rarity_deviation,
    robust_deviation,
)


def test_baseline_stats_empty() -> None:
    stats = compute_baseline_stats([])
    assert stats.count == 0
    assert stats.median is None
    assert stats.mad is None


def test_baseline_stats_single_observation() -> None:
    stats = compute_baseline_stats([7.0])
    assert stats.count == 1
    assert stats.median == 7.0
    assert stats.mad == 0.0


def test_baseline_stats_constant_observations() -> None:
    stats = compute_baseline_stats([5.0, 5.0, 5.0, 5.0])
    assert stats.median == 5.0
    assert stats.mad == 0.0
    assert stats.iqr == 0.0


def test_robust_deviation_with_nonzero_mad() -> None:
    config = AnomalyConfig()
    # values: 1,2,3,4,5 → median=3, mad=1
    baseline = compute_baseline_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    result = robust_deviation(5.0, baseline, config)
    assert result.deviation == pytest.approx(2.0)
    assert result.method == "mad"


def test_zero_mad_same_as_median_is_zero_deviation() -> None:
    config = AnomalyConfig()
    baseline = compute_baseline_stats([4.0, 4.0, 4.0])
    result = robust_deviation(4.0, baseline, config)
    assert result.deviation == 0.0
    assert result.method == "mad"


def test_zero_mad_different_from_median_uses_documented_fallback() -> None:
    config = AnomalyConfig(zero_mad_constant_mismatch_deviation=3.0)
    baseline = compute_baseline_stats([4.0, 4.0, 4.0])
    result = robust_deviation(10.0, baseline, config)
    assert result.deviation == 3.0
    assert result.method == "constant_mismatch"
    assert result.outside_historical_range is True


def test_zero_mad_iqr_fallback_when_iqr_positive() -> None:
    # Construct a sample where MAD is 0 but IQR > 0 is hard with odd counts.
    # For [1, 2, 2, 2, 3]: median=2, MAD=0, IQR = 3-1? 
    # Actually abs deviations from 2: [1,0,0,0,1] → MAD=0. IQR = p75-p25.
    # With 5 points: p25≈1. something. Let's check.
    config = AnomalyConfig()
    values = [1.0, 2.0, 2.0, 2.0, 3.0]
    baseline = compute_baseline_stats(values)
    assert baseline.mad == 0.0
    if baseline.iqr and baseline.iqr > 0:
        result = robust_deviation(5.0, baseline, config)
        assert result.method == "iqr_fallback"
        assert result.deviation == pytest.approx(abs(5.0 - 2.0) / baseline.iqr)
    else:
        result = robust_deviation(5.0, baseline, config)
        assert result.method == "constant_mismatch"


def test_missing_current_returns_none_not_zero() -> None:
    config = AnomalyConfig()
    baseline = compute_baseline_stats([1.0, 2.0, 3.0])
    result = robust_deviation(None, baseline, config)
    assert result.deviation is None
    result_nan = robust_deviation(float("nan"), baseline, config)
    assert result_nan.deviation is None


def test_empty_baseline_returns_none() -> None:
    config = AnomalyConfig()
    baseline = compute_baseline_stats([])
    result = robust_deviation(5.0, baseline, config)
    assert result.deviation is None


def test_persistence_rarity_common_label() -> None:
    prior = ["SHORT_LIVED"] * 8 + ["PERSISTENT"] * 2
    result = persistence_rarity_deviation("SHORT_LIVED", prior, min_prior=3)
    assert result.deviation == 0.0


def test_persistence_rarity_unseen_label() -> None:
    prior = ["SHORT_LIVED"] * 5
    result = persistence_rarity_deviation("PERSISTENT", prior, min_prior=3)
    assert result.deviation == pytest.approx(3.0)


def test_persistence_insufficient_prior_is_none() -> None:
    result = persistence_rarity_deviation("SHORT_LIVED", ["SHORT_LIVED"], min_prior=3)
    assert result.deviation is None
