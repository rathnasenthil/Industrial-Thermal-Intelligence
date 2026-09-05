"""Tests for `src.fingerprinting.monthly_profile`."""

from __future__ import annotations

import pandas as pd
import pytest

from src.fingerprinting.monthly_profile import build_monthly_profile


def _event(event_id: str, facility_id: str | None, month: int, day: int = 1, detection_count: int = 2) -> dict:
    start = f"2023-{month:02d}-{day:02d}T06:00:00+00:00"
    return {"event_id": event_id, "facility_id": facility_id, "event_start": start, "detection_count": detection_count}


def test_monthly_profile_basic_counts() -> None:
    events = pd.DataFrame(
        [
            _event("E1", "F1", month=1, detection_count=3),
            _event("E2", "F1", month=1, detection_count=2),
            _event("E3", "F1", month=2, detection_count=5),
        ]
    )
    profile = build_monthly_profile(events)
    jan = profile.loc[(profile["facility_id"] == "F1") & (profile["month"] == 1)].iloc[0]
    assert jan["event_count"] == 2
    assert jan["detection_count"] == 5
    assert jan["event_fraction"] == pytest.approx(2 / 3)

    feb = profile.loc[(profile["facility_id"] == "F1") & (profile["month"] == 2)].iloc[0]
    assert feb["event_count"] == 1
    assert feb["event_fraction"] == pytest.approx(1 / 3)


def test_monthly_profile_excludes_unassociated_events() -> None:
    events = pd.DataFrame([_event("E1", None, month=1), _event("E2", "F1", month=1)])
    profile = build_monthly_profile(events)
    assert set(profile["facility_id"]) == {"F1"}
    assert profile["event_count"].sum() == 1


def test_monthly_profile_empty_when_no_associations() -> None:
    events = pd.DataFrame([_event("E1", None, month=1)])
    profile = build_monthly_profile(events)
    assert len(profile) == 0
    assert list(profile.columns) == ["facility_id", "month", "event_count", "detection_count", "event_fraction"]


def test_monthly_profile_at_most_twelve_rows_per_facility() -> None:
    events = pd.DataFrame([_event(f"E{m}", "F1", month=m) for m in range(1, 13)] * 2)  # 24 events, 12 months
    profile = build_monthly_profile(events)
    per_facility = profile.loc[profile["facility_id"] == "F1"]
    assert len(per_facility) == 12
    assert set(per_facility["month"]) == set(range(1, 13))


def test_monthly_profile_is_sorted_deterministically() -> None:
    events = pd.DataFrame([_event("E1", "F_z", month=5), _event("E2", "F_a", month=2), _event("E3", "F_a", month=1)])
    profile = build_monthly_profile(events)
    assert list(zip(profile["facility_id"], profile["month"])) == [("F_a", 1), ("F_a", 2), ("F_z", 5)]


def test_monthly_profile_deterministic_across_calls() -> None:
    events = pd.DataFrame([_event(f"E{i}", "F1", month=(i % 12) + 1) for i in range(30)])
    p1 = build_monthly_profile(events)
    p2 = build_monthly_profile(events)
    pd.testing.assert_frame_equal(p1, p2)


def test_missing_required_columns_raise() -> None:
    with pytest.raises(ValueError):
        build_monthly_profile(pd.DataFrame({"event_id": ["E1"]}))
