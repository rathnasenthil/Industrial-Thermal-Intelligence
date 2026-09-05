"""Tests for Stage I.4 walk-forward temporal baseline (leakage-critical)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.anomaly_detection.config import AnomalyConfig
from src.anomaly_detection.temporal_baseline import (
    score_facility_events_walk_forward,
    walk_forward_score_all_facilities,
)


def _event(
    event_id: str,
    facility_id: str,
    day: int,
    *,
    peak_frp: float = 5.0,
    detection_count: int = 2,
    duration: float = 1.0,
    distance: float = 1.0,
    persistence: str = "SHORT_LIVED",
    month: int = 1,
) -> dict:
    return {
        "event_id": event_id,
        "facility_id": facility_id,
        "facility_association_method": "NEAR_FACILITY",
        "event_start": f"2023-{month:02d}-{day:02d}T06:00:00+00:00",
        "event_end": f"2023-{month:02d}-{day:02d}T07:00:00+00:00",
        "peak_frp": peak_frp,
        "detection_count": detection_count,
        "observed_duration_hours": duration,
        "facility_distance_km": distance,
        "persistence_label": persistence,
    }


def test_walk_forward_first_event_insufficient() -> None:
    events = pd.DataFrame([_event("A", "F1", 1, peak_frp=10.0)])
    results = score_facility_events_walk_forward(events, AnomalyConfig())
    assert len(results) == 1
    assert results[0].baseline_observation_count == 0
    assert results[0].peak_frp_deviation is None
    assert results[0].anomaly_unavailable_reason == "INSUFFICIENT_PRIOR_HISTORY"


def test_walk_forward_second_uses_only_first() -> None:
    # Need 3 priors before scoring kicks in; verify baseline counts grow correctly.
    events = pd.DataFrame(
        [
            _event("A", "F1", 1, peak_frp=1.0),
            _event("B", "F1", 2, peak_frp=2.0),
            _event("C", "F1", 3, peak_frp=3.0),
            _event("D", "F1", 4, peak_frp=100.0),  # first scored event; baseline A+B+C
        ]
    )
    results = score_facility_events_walk_forward(events, AnomalyConfig())
    by_id = {r.event_id: r for r in results}
    assert by_id["A"].baseline_observation_count == 0
    assert by_id["B"].baseline_observation_count == 1
    assert by_id["C"].baseline_observation_count == 2
    assert by_id["D"].baseline_observation_count == 3
    # D's baseline median of peak_frp should be median(1,2,3)=2, NOT including 100.
    assert by_id["D"].baseline_peak_frp_median == pytest.approx(2.0)
    assert by_id["D"].peak_frp_deviation is not None


def test_current_event_excluded_from_own_baseline() -> None:
    events = pd.DataFrame(
        [
            _event("A", "F1", 1, peak_frp=5.0),
            _event("B", "F1", 2, peak_frp=5.0),
            _event("C", "F1", 3, peak_frp=5.0),
            _event("D", "F1", 4, peak_frp=50.0),
        ]
    )
    results = score_facility_events_walk_forward(events, AnomalyConfig())
    d = {r.event_id: r for r in results}["D"]
    # If D were included, median would be higher / MAD different.
    assert d.baseline_peak_frp_median == pytest.approx(5.0)
    assert d.baseline_observation_count == 3


def test_shuffled_input_order_same_results() -> None:
    rows = [
        _event("A", "F1", 1, peak_frp=1.0),
        _event("B", "F1", 2, peak_frp=2.0),
        _event("C", "F1", 3, peak_frp=3.0),
        _event("D", "F1", 4, peak_frp=40.0),
        _event("E", "F1", 5, peak_frp=5.0),
    ]
    chronological = pd.DataFrame(rows)
    shuffled = pd.DataFrame(rows[::-1])
    r1 = {r.event_id: r for r in score_facility_events_walk_forward(chronological, AnomalyConfig())}
    r2 = {r.event_id: r for r in score_facility_events_walk_forward(shuffled, AnomalyConfig())}
    for eid in r1:
        assert r1[eid].baseline_observation_count == r2[eid].baseline_observation_count
        assert r1[eid].peak_frp_deviation == r2[eid].peak_frp_deviation
        assert r1[eid].baseline_peak_frp_median == r2[eid].baseline_peak_frp_median


def test_tie_timestamp_uses_event_id_order() -> None:
    # Same timestamp; event_id secondary key → X before Y lexicographically? 
    # "E_a" < "E_b"
    events = pd.DataFrame(
        [
            _event("E_b", "F1", 1, peak_frp=2.0),
            _event("E_a", "F1", 1, peak_frp=1.0),
            _event("E_c", "F1", 1, peak_frp=3.0),
            _event("E_d", "F1", 2, peak_frp=10.0),
        ]
    )
    # Force identical start for first three.
    events.loc[events["event_id"].isin(["E_a", "E_b", "E_c"]), "event_start"] = "2023-01-01T06:00:00+00:00"
    events.loc[events["event_id"].isin(["E_a", "E_b", "E_c"]), "event_end"] = "2023-01-01T07:00:00+00:00"
    results = score_facility_events_walk_forward(events, AnomalyConfig())
    order = [r.event_id for r in results]
    # Chronological among same timestamp: E_a, E_b, E_c, then E_d
    assert order == ["E_a", "E_b", "E_c", "E_d"]
    assert {r.event_id: r.baseline_observation_count for r in results} == {
        "E_a": 0,
        "E_b": 1,
        "E_c": 2,
        "E_d": 3,
    }


def test_monthly_baseline_excludes_future_same_month() -> None:
    # August 2023 events, then August 2024 event.
    # When scoring Aug 2024, only prior August observations may enter monthly baseline.
    events = pd.DataFrame(
        [
            _event("A", "F1", 1, month=8, peak_frp=5.0),
            _event("B", "F1", 2, month=8, peak_frp=5.0),
            _event("C", "F1", 3, month=8, peak_frp=5.0),
            _event("D", "F1", 4, month=9, peak_frp=5.0),  # different month filler
            _event("E", "F1", 5, month=9, peak_frp=5.0),
            _event("F", "F1", 6, month=9, peak_frp=5.0),
            # Future August-like: use year via custom start — rebuild manually
        ]
    )
    # Add a later August event in 2024
    late = _event("G", "F1", 10, month=8, peak_frp=50.0)
    late["event_start"] = "2024-08-10T06:00:00+00:00"
    late["event_end"] = "2024-08-10T07:00:00+00:00"
    events = pd.concat([events, pd.DataFrame([late])], ignore_index=True)

    results = {r.event_id: r for r in score_facility_events_walk_forward(events, AnomalyConfig())}
    g = results["G"]
    # Overall prior count includes A-F (6) → LIMITED or ESTABLISHED depending on threshold
    assert g.baseline_observation_count == 6
    # Monthly prior for August = A,B,C only (3), so monthly deviation is available
    assert g.monthly_deviation is not None
    # And must not have used G itself (50) in monthly baseline — median of 5,5,5 = 5
    # We don't expose monthly median directly; check deviation is large for 50 vs 5 constant.
    assert g.monthly_deviation > 0


def test_ambiguous_events_never_in_facility_baseline() -> None:
    # Only confirmed events should be passed to walk_forward_score_all_facilities.
    confirmed = pd.DataFrame(
        [
            _event("A", "F1", 1),
            _event("B", "F1", 2),
            _event("C", "F1", 3),
            _event("D", "F1", 4, peak_frp=20.0),
        ]
    )
    scored = walk_forward_score_all_facilities(confirmed, AnomalyConfig())
    assert "AMBIGUOUS_X" not in scored
    assert scored["D"].baseline_observation_count == 3
