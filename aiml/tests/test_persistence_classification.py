"""Tests for src.persistence.classification (deterministic persistence/recurrence rules).

Default threshold values mirror the shipped `PersistenceConfig` defaults,
which were calibrated against the real, full 1.17M-detection Stage G run
(see `PersistenceConfig` docstrings for the empirical rationale). The
PERSISTENT rule is an OR of two conditions (high duty cycle, or no long
internal gap) — this matters for several tests below.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.persistence.classification import (
    INSUFFICIENT_OBSERVATIONS,
    PERSISTENT,
    RECURRING,
    SHORT_LIVED,
    classify_events,
    compute_duty_cycle,
    compute_span_days,
)
from src.persistence.config import PersistenceConfig

_BASE_TIME = pd.Timestamp("2023-01-01T06:00:00", tz="UTC")
_DEFAULT_CONFIG = PersistenceConfig(
    min_detections_for_classification=3,
    short_lived_max_duration_hours=48.0,
    persistent_min_duty_cycle=0.85,
    persistent_max_gap_hours=24.0,
)


def _event_row(**overrides) -> dict:
    base = {
        "event_id": "EVT_TEST",
        "detection_count": 5,
        "event_start": _BASE_TIME.isoformat(),
        "event_end": (_BASE_TIME + pd.Timedelta(hours=100)).isoformat(),
        "observed_duration_hours": 100.0,
        "distinct_detection_days": 5,
        "max_gap_hours": 20.0,
    }
    base.update(overrides)
    return base


def _events_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_default_config_matches_calibrated_values() -> None:
    """Sanity-check that the local test fixture matches the shipped defaults."""
    assert _DEFAULT_CONFIG == PersistenceConfig()


def test_insufficient_observations_below_min_detections() -> None:
    df = _events_df([_event_row(detection_count=2)])
    result = classify_events(df, _DEFAULT_CONFIG)
    assert result.iloc[0]["persistence_label"] == INSUFFICIENT_OBSERVATIONS


def test_short_lived_when_duration_below_threshold() -> None:
    df = _events_df(
        [
            _event_row(
                detection_count=3,
                observed_duration_hours=10.0,
                event_end=(_BASE_TIME + pd.Timedelta(hours=10)).isoformat(),
                distinct_detection_days=1,
                max_gap_hours=5.0,
            )
        ]
    )
    result = classify_events(df, _DEFAULT_CONFIG)
    assert result.iloc[0]["persistence_label"] == SHORT_LIVED


def test_persistent_when_high_duty_cycle_and_small_gaps() -> None:
    # 10-calendar-day span (Jan1 06:00 -> Jan10 06:00, i.e. +216h), detected
    # on all 10 of those days, gaps well under threshold -> duty_cycle == 1.0.
    df = _events_df(
        [
            _event_row(
                detection_count=20,
                observed_duration_hours=216.0,
                event_end=(_BASE_TIME + pd.Timedelta(hours=216)).isoformat(),
                distinct_detection_days=10,
                max_gap_hours=20.0,
            )
        ]
    )
    result = classify_events(df, _DEFAULT_CONFIG)
    assert result.iloc[0]["persistence_label"] == PERSISTENT
    assert result.iloc[0]["duty_cycle"] == pytest.approx(1.0)


def test_persistent_via_high_duty_cycle_despite_one_long_gap() -> None:
    """Mirrors the real ~166-day event: near-perfect daily coverage (duty
    close to 1.0) with exactly one longer pause above
    persistent_max_gap_hours. The OR rule must still call this PERSISTENT."""
    df = _events_df(
        [
            _event_row(
                event_id="EVT_LONG_ONE_GAP",
                detection_count=1315,
                observed_duration_hours=166 * 24.0,
                event_end=(_BASE_TIME + pd.Timedelta(days=166)).isoformat(),
                distinct_detection_days=166,
                max_gap_hours=34.9,  # > persistent_max_gap_hours (24), but duty_cycle is very high
            )
        ]
    )
    result = classify_events(df, _DEFAULT_CONFIG)
    assert result.iloc[0]["duty_cycle"] > _DEFAULT_CONFIG.persistent_min_duty_cycle
    assert result.iloc[0]["max_gap_hours"] > _DEFAULT_CONFIG.persistent_max_gap_hours
    assert result.iloc[0]["persistence_label"] == PERSISTENT


def test_persistent_via_short_gap_despite_moderate_duty_cycle() -> None:
    """Other qualifying path of the OR rule: gap stays short even though
    duty cycle alone would not clear the bar."""
    # span_days = 10, distinct_detection_days = 6 -> duty_cycle = 0.6 (< 0.85)
    df = _events_df(
        [
            _event_row(
                detection_count=6,
                observed_duration_hours=216.0,
                event_end=(_BASE_TIME + pd.Timedelta(hours=216)).isoformat(),
                distinct_detection_days=6,
                max_gap_hours=20.0,  # <= persistent_max_gap_hours (24)
            )
        ]
    )
    result = classify_events(df, _DEFAULT_CONFIG)
    assert result.iloc[0]["duty_cycle"] < _DEFAULT_CONFIG.persistent_min_duty_cycle
    assert result.iloc[0]["max_gap_hours"] <= _DEFAULT_CONFIG.persistent_max_gap_hours
    assert result.iloc[0]["persistence_label"] == PERSISTENT


def test_recurring_when_both_duty_cycle_and_gap_fail() -> None:
    """RECURRING requires BOTH signals to fail: low duty cycle AND a long gap."""
    df = _events_df(
        [
            _event_row(
                detection_count=4,
                observed_duration_hours=30 * 24.0,
                event_end=(_BASE_TIME + pd.Timedelta(days=30)).isoformat(),
                distinct_detection_days=4,  # span_days=31 -> duty_cycle ~0.129 (< 0.85)
                max_gap_hours=200.0,  # > 24
            )
        ]
    )
    result = classify_events(df, _DEFAULT_CONFIG)
    assert result.iloc[0]["persistence_label"] == RECURRING


def test_long_duration_high_duty_cycle_event_is_preserved_as_persistent_single_row() -> None:
    """The real ~166-day persistent event (e.g. Jharia-style coal-seam
    fire) must classify as PERSISTENT and remain exactly one row (this
    stage cannot split events since it never touches clustering)."""
    df = _events_df(
        [
            _event_row(
                event_id="EVT_LONG_PERSISTENT",
                detection_count=1315,
                observed_duration_hours=166 * 24.0,
                event_end=(_BASE_TIME + pd.Timedelta(days=166)).isoformat(),
                distinct_detection_days=166,
                max_gap_hours=34.9,
            )
        ]
    )
    result = classify_events(df, _DEFAULT_CONFIG)
    assert len(result) == 1
    assert result.iloc[0]["event_id"] == "EVT_LONG_PERSISTENT"
    assert result.iloc[0]["persistence_label"] == PERSISTENT


def test_boundary_duration_exactly_at_short_lived_threshold_is_short_lived() -> None:
    df = _events_df(
        [
            _event_row(
                detection_count=3,
                observed_duration_hours=48.0,
                event_end=(_BASE_TIME + pd.Timedelta(hours=48)).isoformat(),
                distinct_detection_days=2,
                max_gap_hours=24.0,
            )
        ]
    )
    result = classify_events(df, _DEFAULT_CONFIG)
    assert result.iloc[0]["persistence_label"] == SHORT_LIVED


def test_boundary_gap_exactly_at_persistent_threshold_is_persistent() -> None:
    # duty_cycle deliberately low (0.6, < 0.85) so only the gap condition can qualify.
    df = _events_df(
        [
            _event_row(
                detection_count=6,
                observed_duration_hours=216.0,
                event_end=(_BASE_TIME + pd.Timedelta(hours=216)).isoformat(),
                distinct_detection_days=6,
                max_gap_hours=24.0,  # exactly at persistent_max_gap_hours
            )
        ]
    )
    result = classify_events(df, _DEFAULT_CONFIG)
    assert result.iloc[0]["duty_cycle"] == pytest.approx(0.6)
    assert result.iloc[0]["persistence_label"] == PERSISTENT


def test_boundary_duty_cycle_exactly_at_persistent_threshold_is_persistent() -> None:
    # span_days=20, distinct_detection_days=17 -> duty_cycle exactly 0.85.
    df = _events_df(
        [
            _event_row(
                detection_count=17,
                observed_duration_hours=19 * 24.0 + 1,
                event_end=(_BASE_TIME + pd.Timedelta(days=19, hours=1)).isoformat(),
                distinct_detection_days=17,
                max_gap_hours=48.0,  # well above persistent_max_gap_hours, so only duty_cycle can qualify
            )
        ]
    )
    result = classify_events(df, _DEFAULT_CONFIG)
    assert result.iloc[0]["duty_cycle"] == pytest.approx(0.85)
    assert result.iloc[0]["persistence_label"] == PERSISTENT


def test_compute_span_days_counts_calendar_days_inclusively() -> None:
    start = pd.Series([pd.Timestamp("2023-01-01T23:50:00", tz="UTC")])
    end = pd.Series([pd.Timestamp("2023-01-02T00:10:00", tz="UTC")])
    span = compute_span_days(start, end)
    assert span.iloc[0] == 2


def test_compute_span_days_same_day_is_one() -> None:
    start = pd.Series([pd.Timestamp("2023-01-01T01:00:00", tz="UTC")])
    end = pd.Series([pd.Timestamp("2023-01-01T20:00:00", tz="UTC")])
    assert compute_span_days(start, end).iloc[0] == 1


def test_compute_duty_cycle_clips_at_one() -> None:
    distinct_days = pd.Series([5])
    span_days = pd.Series([5])
    assert compute_duty_cycle(distinct_days, span_days).iloc[0] == pytest.approx(1.0)


def test_missing_required_column_raises() -> None:
    df = pd.DataFrame([_event_row()]).drop(columns=["max_gap_hours"])
    with pytest.raises(ValueError):
        classify_events(df, _DEFAULT_CONFIG)


def test_persistence_basis_is_populated_for_every_label() -> None:
    df = _events_df(
        [
            _event_row(event_id="A", detection_count=2),
            _event_row(
                event_id="B",
                detection_count=3,
                observed_duration_hours=5.0,
                event_end=(_BASE_TIME + pd.Timedelta(hours=5)).isoformat(),
                distinct_detection_days=1,
                max_gap_hours=5.0,
            ),
            _event_row(
                event_id="C",
                detection_count=10,
                observed_duration_hours=240.0,
                event_end=(_BASE_TIME + pd.Timedelta(hours=240)).isoformat(),
                distinct_detection_days=10,
                max_gap_hours=20.0,
            ),
            _event_row(
                event_id="D",
                detection_count=4,
                observed_duration_hours=720.0,
                event_end=(_BASE_TIME + pd.Timedelta(hours=720)).isoformat(),
                distinct_detection_days=4,
                max_gap_hours=200.0,
            ),
        ]
    )
    result = classify_events(df, _DEFAULT_CONFIG)
    assert (result["persistence_basis"] != "").all()
    assert result["persistence_basis"].notna().all()
    labels = set(result["persistence_label"])
    assert labels == {INSUFFICIENT_OBSERVATIONS, SHORT_LIVED, PERSISTENT, RECURRING}
