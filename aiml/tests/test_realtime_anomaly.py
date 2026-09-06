"""
Phase 7: realtime I.4 must match batch walk-forward semantics for one event.

Does not invoke ``run_anomaly_detection()`` over a full events table.
Does not use I.3 fingerprint tables as the scoring baseline.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from realtime.anomaly import (
    process_event_anomaly,
    unavailable_anomaly_result,
)
from src.anomaly_detection.anomaly_scoring import compute_anomaly_score
from src.anomaly_detection.config import (
    ANOMALOUS,
    ELEVATED,
    INSUFFICIENT_HISTORY,
    NORMAL,
    REASON_AMBIGUOUS,
    REASON_INSUFFICIENT_PRIOR,
    REASON_NO_FACILITY,
    AnomalyConfig,
    HISTORY_ESTABLISHED,
    HISTORY_INSUFFICIENT,
    HISTORY_LIMITED,
    HISTORY_NONE,
)
from src.anomaly_detection.temporal_baseline import score_facility_events_walk_forward


def _row(
    event_id: str,
    facility_id: str,
    *,
    day: int,
    month: int = 1,
    peak_frp: float = 5.0,
    detection_count: int = 2,
    duration: float = 1.0,
    distance: float = 1.0,
    persistence: str = "SHORT_LIVED",
    method: str = "NEAR_FACILITY",
) -> dict:
    start = f"2023-{month:02d}-{day:02d}T06:00:00+00:00"
    return {
        "event_id": event_id,
        "event_start": start,
        "event_end": start,
        "peak_frp": peak_frp,
        "detection_count": detection_count,
        "observed_duration_hours": duration,
        "persistence_label": persistence,
        "facility_id": facility_id,
        "facility_association_method": method,
        "facility_distance_km": distance,
    }


def _facility_df(n_prior: int, *, current_id: str = "CUR", current_overrides: dict | None = None) -> pd.DataFrame:
    rows = []
    for i in range(1, n_prior + 1):
        rows.append(
            _row(
                f"P{i:02d}",
                "F1",
                day=min(i, 28),
                month=((i - 1) % 12) + 1,
                peak_frp=5.0,
            )
        )
    cur = _row(current_id, "F1", day=min(n_prior + 1, 28), month=((n_prior) % 12) + 1)
    if current_overrides:
        cur.update(current_overrides)
    rows.append(cur)
    return pd.DataFrame(rows)


def _batch_current(df: pd.DataFrame, event_id: str, config: AnomalyConfig | None = None):
    cfg = config or AnomalyConfig()
    scored = score_facility_events_walk_forward(df, cfg)
    inputs = next(s for s in scored if s.event_id == event_id)
    score = compute_anomaly_score(inputs, cfg)
    return inputs, score


@pytest.mark.parametrize(
    "n_prior,history,status",
    [
        (0, HISTORY_NONE, INSUFFICIENT_HISTORY),
        (1, HISTORY_INSUFFICIENT, INSUFFICIENT_HISTORY),
        (2, HISTORY_INSUFFICIENT, INSUFFICIENT_HISTORY),
        (3, HISTORY_LIMITED, None),  # status depends on score
        (9, HISTORY_LIMITED, None),
        (10, HISTORY_ESTABLISHED, None),
    ],
)
def test_prior_history_thresholds(n_prior: int, history: str, status: str | None) -> None:
    df = _facility_df(n_prior)
    result = process_event_anomaly(df, "CUR")
    assert result.baseline_observation_count == n_prior
    assert result.baseline_history_status == history
    if n_prior < 3:
        assert result.anomaly_status == INSUFFICIENT_HISTORY
        assert result.anomaly_unavailable_reason == REASON_INSUFFICIENT_PRIOR
        assert result.anomaly_score is None
        assert result.peak_frp_deviation is None
    else:
        assert result.anomaly_unavailable_reason is None
        assert result.anomaly_score is not None
        if status is not None:
            assert result.anomaly_status == status


def test_current_event_excluded_from_own_baseline() -> None:
    # Current has extreme FRP; if self-included, median would shift and deviation shrink.
    rows = [_row(f"P{i}", "F1", day=i, peak_frp=5.0) for i in range(1, 6)]
    rows.append(_row("CUR", "F1", day=10, peak_frp=100.0))
    df = pd.DataFrame(rows)
    result = process_event_anomaly(df, "CUR")
    assert result.baseline_observation_count == 5
    assert result.peak_frp_deviation is not None
    assert result.peak_frp_deviation > 0
    # Baseline median implied: deviation = |100-5|/MAD; MAD of five 5.0s is 0 → constant mismatch 3.0
    assert result.peak_frp_deviation == pytest.approx(3.0)


def test_ordering_by_event_start_then_event_id() -> None:
    # Same start time: event_id order decides who is prior.
    rows = [
        _row("B", "F1", day=1, peak_frp=5.0),
        _row("A", "F1", day=1, peak_frp=5.0),
        _row("C", "F1", day=1, peak_frp=50.0),
    ]
    # Force identical timestamps
    for r in rows:
        r["event_start"] = "2023-01-01T06:00:00+00:00"
        r["event_end"] = r["event_start"]
    df = pd.DataFrame(rows)
    result = process_event_anomaly(df, "C")
    # Sorted A, B, C → two priors
    assert result.baseline_observation_count == 2
    assert result.anomaly_status == INSUFFICIENT_HISTORY


def test_peak_frp_event_size_duration_distance_deviations() -> None:
    rows = [
        _row("P1", "F1", day=1, peak_frp=5.0, detection_count=2, duration=1.0, distance=1.0),
        _row("P2", "F1", day=2, peak_frp=5.0, detection_count=2, duration=1.0, distance=1.0),
        _row("P3", "F1", day=3, peak_frp=5.0, detection_count=2, duration=1.0, distance=1.0),
        _row(
            "CUR",
            "F1",
            day=4,
            peak_frp=20.0,
            detection_count=10,
            duration=8.0,
            distance=4.0,
        ),
    ]
    df = pd.DataFrame(rows)
    result = process_event_anomaly(df, "CUR")
    assert result.peak_frp_deviation is not None and result.peak_frp_deviation > 0
    assert result.event_size_deviation is not None and result.event_size_deviation > 0
    assert result.duration_deviation is not None and result.duration_deviation > 0
    assert result.distance_deviation is not None and result.distance_deviation > 0


def test_persistence_rarity() -> None:
    rows = [
        _row("P1", "F1", day=1, persistence="SHORT_LIVED"),
        _row("P2", "F1", day=2, persistence="SHORT_LIVED"),
        _row("P3", "F1", day=3, persistence="SHORT_LIVED"),
        _row("CUR", "F1", day=4, persistence="PERSISTENT"),
    ]
    df = pd.DataFrame(rows)
    result = process_event_anomaly(df, "CUR")
    # Never seen among priors → rarity deviation 3.0
    assert result.persistence_deviation == pytest.approx(3.0)


def test_monthly_requires_three_same_month_priors() -> None:
    # Two January priors — monthly should be None; three — computed.
    rows = [
        _row("P1", "F1", day=1, month=1, peak_frp=5.0),
        _row("P2", "F1", day=2, month=1, peak_frp=5.0),
        _row("CUR", "F1", day=15, month=1, peak_frp=50.0),
    ]
    r2 = process_event_anomaly(pd.DataFrame(rows), "CUR")
    assert r2.monthly_deviation is None  # only 2 same-month priors; also <3 history

    rows.insert(2, _row("P3", "F1", day=3, month=1, peak_frp=5.0))
    # Need another prior for LIMITED history so monthly can populate when month_priors>=3
    rows.insert(3, _row("P4", "F1", day=4, month=2, peak_frp=5.0))  # different month
    # Rebuild: P1,P2,P3 Jan + P4 Feb + CUR Jan → Jan priors before CUR = 3
    rows = [
        _row("P1", "F1", day=1, month=1, peak_frp=5.0),
        _row("P2", "F1", day=2, month=1, peak_frp=5.0),
        _row("P3", "F1", day=3, month=1, peak_frp=5.0),
        _row("P4", "F1", day=4, month=2, peak_frp=5.0),
        _row("CUR", "F1", day=15, month=1, peak_frp=50.0),
    ]
    r3 = process_event_anomaly(pd.DataFrame(rows), "CUR")
    # CUR is Jan 15; Feb prior sorts after CUR, so only 3 chronological priors.
    assert r3.baseline_observation_count == 3
    assert r3.monthly_deviation is not None
    assert r3.monthly_deviation > 0


def test_missing_feature_excluded_not_zero() -> None:
    rows = [
        _row("P1", "F1", day=1, peak_frp=5.0),
        _row("P2", "F1", day=2, peak_frp=5.0),
        _row("P3", "F1", day=3, peak_frp=5.0),
        _row("CUR", "F1", day=4, peak_frp=np.nan),
    ]
    df = pd.DataFrame(rows)
    result = process_event_anomaly(df, "CUR")
    assert result.peak_frp_deviation is None
    # Other features may still score
    assert result.anomaly_score is not None or result.features_evaluated >= 0


def test_zero_mad_constant_mismatch() -> None:
    rows = [_row(f"P{i}", "F1", day=i, peak_frp=5.0) for i in range(1, 4)]
    rows.append(_row("CUR", "F1", day=5, peak_frp=9.0))
    result = process_event_anomaly(pd.DataFrame(rows), "CUR")
    assert result.peak_frp_deviation == pytest.approx(3.0)


def test_batch_realtime_score_and_status_parity() -> None:
    rows = [_row(f"P{i}", "F1", day=i, peak_frp=5.0 + (i % 3)) for i in range(1, 12)]
    rows.append(_row("CUR", "F1", day=20, peak_frp=80.0, detection_count=40, duration=12.0))
    df = pd.DataFrame(rows)
    rt = process_event_anomaly(df, "CUR")
    inputs, score = _batch_current(df, "CUR")
    assert rt.anomaly_score == pytest.approx(score.anomaly_score)
    assert rt.anomaly_status == score.anomaly_status
    assert rt.baseline_observation_count == inputs.baseline_observation_count
    assert rt.peak_frp_deviation == pytest.approx(inputs.peak_frp_deviation)
    assert rt.anomaly_confidence == score.anomaly_confidence


def test_ambiguous_unavailable() -> None:
    result = unavailable_anomaly_result("E_AMB", reason=REASON_AMBIGUOUS)
    assert result.baseline_history_status == "NOT_APPLICABLE"
    assert result.anomaly_unavailable_reason == REASON_AMBIGUOUS
    assert result.anomaly_score is None
    assert result.anomaly_status == INSUFFICIENT_HISTORY
    assert "AMBIGUOUS" in result.anomaly_explanation


def test_no_facility_unavailable() -> None:
    result = unavailable_anomaly_result("E_NONE", reason=REASON_NO_FACILITY)
    assert result.baseline_history_status == "NOT_APPLICABLE"
    assert result.anomaly_unavailable_reason == REASON_NO_FACILITY
    assert result.anomaly_score is None
    assert result.anomaly_status == INSUFFICIENT_HISTORY


def test_does_not_use_fingerprint_tables_or_full_pipeline() -> None:
    df = _facility_df(5)
    with (
        patch("src.anomaly_detection.anomaly_pipeline.run_anomaly_detection") as full,
        patch("src.anomaly_detection.anomaly_pipeline.load_fingerprints") as load_fp,
    ):
        process_event_anomaly(df, "CUR")
        full.assert_not_called()
        load_fp.assert_not_called()


def test_normal_elevated_anomalous_bands() -> None:
    # Established baseline of near-identical events; mild vs extreme current.
    base = [_row(f"P{i}", "F1", day=i, peak_frp=5.0, detection_count=2, duration=1.0) for i in range(1, 11)]
    mild = list(base) + [_row("CUR", "F1", day=15, peak_frp=5.0, detection_count=2, duration=1.0)]
    extreme = list(base) + [
        _row("CUR", "F1", day=15, peak_frp=500.0, detection_count=200, duration=100.0, distance=20.0)
    ]
    r_mild = process_event_anomaly(pd.DataFrame(mild), "CUR")
    r_ext = process_event_anomaly(pd.DataFrame(extreme), "CUR")
    assert r_mild.anomaly_status == NORMAL
    assert r_ext.anomaly_status in (ELEVATED, ANOMALOUS)
    assert r_ext.anomaly_score is not None and r_ext.anomaly_score >= 2.0
