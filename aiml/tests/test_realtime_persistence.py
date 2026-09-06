"""
Phase 4: realtime G.1 persistence must match batch ``classify_events``.

Does not invoke ``run_persistence_characterization()`` over a full events table.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from realtime.persistence import (
    batch_pipeline_invocation_count,
    process_event_persistence,
    process_event_persistence_from_mapping,
)
from src.event_formation.event_features import compute_event_row
from src.persistence.classification import (
    INSUFFICIENT_OBSERVATIONS,
    PERSISTENT,
    RECURRING,
    SHORT_LIVED,
    classify_events,
)
from src.persistence.config import PersistenceConfig
from src.persistence.persistence_pipeline import run_persistence_characterization

_BASE = datetime(2023, 1, 1, 6, 0, 0, tzinfo=timezone.utc)
_CFG = PersistenceConfig()


def _batch_from_times(event_id: str, times: list[datetime]) -> dict:
    """Build Stage G temporal row + classify via batch path."""
    n = len(times)
    detections = pd.DataFrame(
        {
            "latitude": [20.0] * n,
            "longitude": [85.0] * n,
            "acq_datetime": pd.to_datetime(times, utc=True),
            "frp": [1.0] * n,
            "frp_valid": [True] * n,
            "bright_ti4": [330.0] * n,
            "bright_ti5": [290.0] * n,
            "confidence": ["n"] * n,
            "daynight": ["D"] * n,
        }
    )
    row = compute_event_row(event_id, detections)
    events_df = pd.DataFrame(
        [
            {
                "event_id": event_id,
                "detection_count": row["detection_count"],
                "event_start": row["event_start"],
                "event_end": row["event_end"],
                "observed_duration_hours": row["observed_duration_hours"],
                "distinct_detection_days": row["distinct_detection_days"],
                "max_gap_hours": row["max_gap_hours"],
            }
        ]
    )
    classified = classify_events(events_df, _CFG).iloc[0]
    return {
        "detection_count": int(classified["detection_count"]),
        "distinct_detection_days": int(classified["distinct_detection_days"]),
        "span_days": int(classified["span_days"]),
        "observed_duration_hours": float(classified["observed_duration_hours"]),
        "duty_cycle": float(classified["duty_cycle"]),
        "mean_gap_hours": row["mean_gap_hours"],
        "max_gap_hours": float(classified["max_gap_hours"])
        if pd.notna(classified["max_gap_hours"])
        else float("nan"),
        "persistence_label": str(classified["persistence_label"]),
    }


def _assert_parity(times: list[datetime], event_id: str = "EVT_PARITY") -> None:
    rt = process_event_persistence(event_id, times, config=_CFG)
    batch = _batch_from_times(event_id, times)
    assert rt.detection_count == batch["detection_count"]
    assert rt.distinct_detection_days == batch["distinct_detection_days"]
    assert rt.span_days == batch["span_days"]
    assert rt.observed_duration_hours == pytest.approx(batch["observed_duration_hours"])
    assert rt.duty_cycle == pytest.approx(batch["duty_cycle"])
    assert rt.persistence_label == batch["persistence_label"]
    if times and len(times) == 1:
        assert rt.mean_gap_hours is None
        assert rt.max_gap_hours is None
    else:
        assert rt.mean_gap_hours == pytest.approx(batch["mean_gap_hours"])
        assert rt.max_gap_hours == pytest.approx(batch["max_gap_hours"])


def test_one_observation_insufficient() -> None:
    times = [_BASE]
    feat = process_event_persistence("EVT_1", times)
    assert feat.detection_count == 1
    assert feat.distinct_detection_days == 1
    assert feat.span_days == 1
    assert feat.observed_duration_hours == pytest.approx(0.0)
    assert feat.duty_cycle == pytest.approx(1.0)
    assert feat.mean_gap_hours is None
    assert feat.max_gap_hours is None
    assert feat.persistence_label == INSUFFICIENT_OBSERVATIONS
    _assert_parity(times, "EVT_1")


def test_two_observations_different_timestamps() -> None:
    times = [_BASE, _BASE + timedelta(hours=12)]
    feat = process_event_persistence("EVT_2", times)
    assert feat.detection_count == 2
    assert feat.persistence_label == INSUFFICIENT_OBSERVATIONS
    assert feat.mean_gap_hours == pytest.approx(12.0)
    assert feat.max_gap_hours == pytest.approx(12.0)
    _assert_parity(times, "EVT_2")


def test_same_day_observations() -> None:
    times = [
        _BASE,
        _BASE + timedelta(hours=2),
        _BASE + timedelta(hours=4),
    ]
    feat = process_event_persistence("EVT_SAME", times)
    assert feat.distinct_detection_days == 1
    assert feat.span_days == 1
    assert feat.duty_cycle == pytest.approx(1.0)
    assert feat.persistence_label == SHORT_LIVED
    _assert_parity(times, "EVT_SAME")


def test_multiple_days_short_lived() -> None:
    times = [
        _BASE,
        _BASE + timedelta(hours=24),
        _BASE + timedelta(hours=40),
    ]
    feat = process_event_persistence("EVT_SL", times)
    assert feat.detection_count == 3
    assert feat.distinct_detection_days == 2
    assert feat.observed_duration_hours == pytest.approx(40.0)
    assert feat.persistence_label == SHORT_LIVED
    _assert_parity(times, "EVT_SL")


def test_persistent_high_duty_cycle() -> None:
    # 10 calendar days, daily detection → duty_cycle = 1.0, duration > 48h
    times = [_BASE + timedelta(days=i) for i in range(10)]
    feat = process_event_persistence("EVT_P", times)
    assert feat.detection_count == 10
    assert feat.distinct_detection_days == 10
    assert feat.span_days == 10
    assert feat.duty_cycle == pytest.approx(1.0)
    assert feat.persistence_label == PERSISTENT
    _assert_parity(times, "EVT_P")


def test_recurring_low_duty_and_long_gap() -> None:
    # Sparse detections over 30 days with large gaps — but wait: realtime events
    # use 36h continuity so such a set wouldn't form one event in Phase 3.
    # G.1 still classifies the detection set independently (batch parity).
    times = [
        _BASE,
        _BASE + timedelta(days=10),
        _BASE + timedelta(days=20),
        _BASE + timedelta(days=30),
    ]
    feat = process_event_persistence("EVT_R", times)
    assert feat.persistence_label == RECURRING
    _assert_parity(times, "EVT_R")


def test_insufficient_below_min_detections() -> None:
    times = [_BASE, _BASE + timedelta(hours=10)]
    feat = process_event_persistence("EVT_INSUF", times)
    assert feat.persistence_label == INSUFFICIENT_OBSERVATIONS
    _assert_parity(times, "EVT_INSUF")


def test_null_timestamps_ignored() -> None:
    times = [_BASE, None, _BASE + timedelta(hours=5)]  # type: ignore[list-item]
    feat = process_event_persistence("EVT_NULL", times)
    assert feat.detection_count == 2
    assert feat.persistence_label == INSUFFICIENT_OBSERVATIONS


def test_all_invalid_timestamps_raise() -> None:
    with pytest.raises(ValueError, match="no valid detection"):
        process_event_persistence("EVT_BAD", [None, None])  # type: ignore[list-item]


def test_idempotent_reprocess() -> None:
    times = [_BASE + timedelta(days=i) for i in range(5)]
    a = process_event_persistence("EVT_IDEM", times)
    b = process_event_persistence("EVT_IDEM", times)
    assert a.to_dict() == b.to_dict()


def test_mapping_wrapper() -> None:
    feat = process_event_persistence_from_mapping(
        {"event_id": "EVT_MAP"},
        [
            {"acq_datetime": _BASE},
            {"acq_datetime": _BASE + timedelta(hours=1)},
        ],
    )
    assert feat.event_id == "EVT_MAP"
    assert feat.detection_count == 2


def test_batch_fixture_persistent_parity() -> None:
    """Known batch classification fixture → realtime must agree on labels."""
    # Mirrors test_persistent_when_high_duty_cycle_and_small_gaps detection pattern.
    times = [_BASE + timedelta(hours=24 * i) for i in range(10)]
    # Add denser same-day duplicates to raise detection_count like the fixture.
    times += [_BASE + timedelta(hours=24 * i + 1) for i in range(10)]
    _assert_parity(times, "EVT_FIXTURE_P")


def test_realtime_does_not_call_batch_orchestrator() -> None:
    """Instrumentation: full-table G.1 must never run on the realtime path."""
    before = batch_pipeline_invocation_count()

    def _boom(*_a, **_k):
        raise AssertionError("run_persistence_characterization must not be called")

    with patch(
        "src.persistence.persistence_pipeline.run_persistence_characterization",
        side_effect=_boom,
    ):
        # Also ensure direct import isn't used by process_event_persistence.
        process_event_persistence("EVT_X", [_BASE, _BASE + timedelta(hours=3)])

    assert batch_pipeline_invocation_count() == before
    # Sanity: the batch function still exists and works when called intentionally.
    df = pd.DataFrame(
        [
            {
                "event_id": "EVT_BATCH",
                "detection_count": 5,
                "event_start": _BASE.isoformat(),
                "event_end": (_BASE + timedelta(hours=100)).isoformat(),
                "observed_duration_hours": 100.0,
                "distinct_detection_days": 5,
                "max_gap_hours": 20.0,
            }
        ]
    )
    result = run_persistence_characterization(df, _CFG)
    assert len(result.events_df) == 1
