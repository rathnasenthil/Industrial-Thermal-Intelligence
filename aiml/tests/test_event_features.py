"""Tests for src.event_formation.event_features (event-level aggregation)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.event_formation.event_features import build_thermal_events, compute_event_row

_BASE_TIME = pd.Timestamp("2023-01-01T06:00:00", tz="UTC")


def _detections(**overrides) -> pd.DataFrame:
    n = 4
    base = {
        "latitude": [21.50, 21.501, 21.502, 21.503],
        "longitude": [82.10, 82.101, 82.102, 82.103],
        "acq_datetime": [_BASE_TIME, _BASE_TIME + pd.Timedelta(hours=12), _BASE_TIME + pd.Timedelta(hours=24), _BASE_TIME + pd.Timedelta(hours=36)],
        "frp": [2.5, 5.0, np.nan, 10.0],
        "frp_valid": [True, True, False, True],
        "bright_ti4": [330.0, 335.0, 328.0, 340.0],
        "bright_ti5": [290.0, 295.0, 288.0, 300.0],
        "confidence": ["n", "n", "l", "h"],
        "daynight": ["D", "N", "D", "N"],
        "event_id": ["EVT_0000001"] * n,
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_detection_count_and_duration() -> None:
    row = compute_event_row("EVT_0000001", _detections())
    assert row["detection_count"] == 4
    assert row["observed_duration_hours"] == pytest.approx(36.0)
    assert row["event_start"] == _BASE_TIME.isoformat()
    assert row["event_end"] == (_BASE_TIME + pd.Timedelta(hours=36)).isoformat()


def test_frp_statistics_exclude_invalid_and_do_not_fabricate() -> None:
    row = compute_event_row("EVT_0000001", _detections())

    # Only the 3 valid FRP values (2.5, 5.0, 10.0) should be used.
    assert row["frp_valid_count"] == 3
    assert row["peak_frp"] == pytest.approx(10.0)
    assert row["mean_frp"] == pytest.approx((2.5 + 5.0 + 10.0) / 3)
    assert row["median_frp"] == pytest.approx(5.0)
    assert row["total_frp"] == pytest.approx(17.5)


def test_frp_statistics_are_none_when_all_frp_invalid() -> None:
    df = _detections(frp=[np.nan, np.nan, np.nan, np.nan], frp_valid=[False, False, False, False])
    row = compute_event_row("EVT_0000001", df)

    assert row["frp_valid_count"] == 0
    assert row["peak_frp"] is None
    assert row["mean_frp"] is None
    assert row["total_frp"] is None


def test_confidence_counts_preserve_native_codes() -> None:
    row = compute_event_row("EVT_0000001", _detections())
    assert row["confidence_n_count"] == 2
    assert row["confidence_l_count"] == 1
    assert row["confidence_h_count"] == 1
    assert row["confidence_distribution"] == {"n": 2, "l": 1, "h": 1}


def test_daynight_counts() -> None:
    row = compute_event_row("EVT_0000001", _detections())
    assert row["day_detection_count"] == 2
    assert row["night_detection_count"] == 2


def test_bright_ti4_ti5_stats() -> None:
    row = compute_event_row("EVT_0000001", _detections())
    assert row["max_bright_ti4"] == pytest.approx(340.0)
    assert row["mean_bright_ti4"] == pytest.approx((330 + 335 + 328 + 340) / 4)
    assert row["max_bright_ti5"] == pytest.approx(300.0)


def test_persistence_fields() -> None:
    row = compute_event_row("EVT_0000001", _detections())
    assert row["distinct_detection_days"] >= 1
    assert row["mean_gap_hours"] == pytest.approx(12.0)
    assert row["max_gap_hours"] == pytest.approx(12.0)


def test_build_thermal_events_groups_by_event_id_and_sorts_by_start() -> None:
    df1 = _detections(event_id=["EVT_0000002"] * 4, acq_datetime=[_BASE_TIME + pd.Timedelta(hours=100)] * 4)
    df2 = _detections(event_id=["EVT_0000001"] * 4)
    combined = pd.concat([df1, df2], ignore_index=True)

    events_df = build_thermal_events(combined)

    assert len(events_df) == 2
    assert list(events_df["event_id"]) == ["EVT_0000001", "EVT_0000002"]


def test_missing_required_column_raises() -> None:
    df = _detections().drop(columns=["frp"])
    with pytest.raises(ValueError):
        compute_event_row("EVT_0000001", df)
