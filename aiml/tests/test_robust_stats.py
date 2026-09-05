"""Tests for `src.fingerprinting.robust_stats`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fingerprinting.robust_stats import grouped_summary_stats, mad, median, quantile, summary_stats


def test_median_basic() -> None:
    assert median([1, 2, 3, 4, 5]) == 3
    assert median([1, 2, 3, 4]) == 2.5


def test_median_empty_is_none() -> None:
    assert median([]) is None
    assert median([np.nan, np.nan]) is None


def test_median_single_observation() -> None:
    assert median([7.0]) == 7.0


def test_median_drops_nan() -> None:
    assert median([1.0, np.nan, 3.0]) == 2.0


def test_mad_basic() -> None:
    # median = 3; |x-3| = [2,1,0,1,2]; median of that = 1
    assert mad([1, 2, 3, 4, 5]) == 1.0


def test_mad_single_observation_is_zero() -> None:
    assert mad([42.0]) == 0.0


def test_mad_empty_is_none() -> None:
    assert mad([]) is None


def test_mad_is_not_scaled_to_normal_consistent_std() -> None:
    # For a large sample from N(0,1), raw MAD ~= 0.6745 * std, NOT std
    # itself -- this test locks in that this module never applies the
    # ~1.4826 scale factor implicitly.
    rng = np.random.default_rng(42)
    sample = rng.normal(loc=0.0, scale=10.0, size=5000)
    raw_mad = mad(sample)
    assert raw_mad is not None
    assert 5.0 < raw_mad < 8.0  # well below the ~10 std, not scaled up to match it


def test_quantile_basic() -> None:
    values = list(range(1, 101))  # 1..100
    assert quantile(values, 0.5) == pytest.approx(50.5)
    assert quantile(values, 0.25) == pytest.approx(25.75)
    assert quantile(values, 0.90) == pytest.approx(90.1)


def test_quantile_empty_is_none() -> None:
    assert quantile([], 0.5) is None


def test_extreme_value_does_not_dominate_median() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 100000.0]
    # Median of 6 values is the mean of the middle two (3, 4) -- barely
    # moved by the extreme outlier, unlike the arithmetic mean would be.
    assert median(values) == 3.5
    assert mad(values) < 5.0


def test_extreme_value_is_preserved_in_max() -> None:
    stats = summary_stats([1.0, 2.0, 3.0, 100000.0], prefix="frp")
    assert stats["frp_max"] == 100000.0


def test_summary_stats_all_none_for_empty_input() -> None:
    stats = summary_stats([], prefix="frp")
    for key, value in stats.items():
        assert value is None, f"{key} should be None, got {value}"


def test_summary_stats_all_none_for_all_nan_input() -> None:
    stats = summary_stats([np.nan, np.nan, np.nan], prefix="frp")
    for key, value in stats.items():
        assert value is None, f"{key} should be None, got {value}"


def test_missing_frp_does_not_become_zero() -> None:
    # A facility with only missing FRP observations must never report 0.
    stats = summary_stats([np.nan, np.nan], prefix="peak_frp")
    assert stats["peak_frp_median"] is None
    assert stats["peak_frp_max"] is None


def test_grouped_summary_stats_matches_scalar_functions() -> None:
    df = pd.DataFrame({"facility_id": ["A", "A", "A", "B", "B"], "value": [1.0, 2.0, 3.0, 10.0, 20.0]})
    grouped = grouped_summary_stats(df, "facility_id", "value", "x")
    assert grouped.loc["A", "x_median"] == median([1.0, 2.0, 3.0])
    assert grouped.loc["A", "x_mad"] == mad([1.0, 2.0, 3.0])
    assert grouped.loc["B", "x_max"] == 20.0


def test_grouped_summary_stats_all_nan_group_stays_null() -> None:
    df = pd.DataFrame({"facility_id": ["A", "A"], "value": [np.nan, np.nan]})
    grouped = grouped_summary_stats(df, "facility_id", "value", "x")
    assert pd.isna(grouped.loc["A", "x_median"])
    assert pd.isna(grouped.loc["A", "x_max"])
