"""
Phase 6: realtime I.3 must match batch build_facility_fingerprints /
build_monthly_profile semantics for a single facility.

Does not invoke ``run_facility_fingerprinting()`` over the full facility
universe.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from realtime.facility_fingerprint import (
    CONFIRMED_ASSOCIATION_METHODS,
    process_facility_fingerprint,
)
from src.fingerprinting.facility_fingerprint import build_facility_fingerprints
from src.fingerprinting.fingerprint_config import (
    ESTABLISHED_BASELINE,
    FingerprintConfig,
    INSUFFICIENT_HISTORY,
    LIMITED_HISTORY,
    NO_OBSERVATIONS,
)
from src.fingerprinting.monthly_profile import build_monthly_profile

_BASE_EVENT = {
    "detection_count": 2,
    "distinct_detection_days": 1,
    "observed_duration_hours": 1.0,
    "day_detection_count": 2,
    "night_detection_count": 0,
    "persistence_label": "SHORT_LIVED",
    "facility_association_method": "NEAR_FACILITY",
    "facility_attribution_confidence": "MEDIUM",
    "facility_distance_km": 1.0,
    "candidate_facility_ids": "",
}


def _event(event_id: str, facility_id: str | None, month: int, day: int = 1, **overrides) -> dict:
    row = dict(_BASE_EVENT)
    row["event_id"] = event_id
    row["facility_id"] = facility_id
    row["event_start"] = f"2023-{month:02d}-{day:02d}T06:00:00+00:00"
    row["event_end"] = row["event_start"]
    row["peak_frp"] = overrides.pop("peak_frp", 5.0)
    row.update(overrides)
    return row


def _facility(fid: str = "F1", name: str = "Fac1", ftype: str = "MINE") -> dict:
    return {"facility_id": fid, "facility_name": name, "facility_type": ftype}


def _rt(facility: dict, events: list[dict]):
    return process_facility_fingerprint(facility, pd.DataFrame(events), config=FingerprintConfig())


def _batch_one(facility: dict, events: list[dict]):
    facilities = pd.DataFrame([facility])
    fp = build_facility_fingerprints(pd.DataFrame(events), facilities, FingerprintConfig())
    monthly = build_monthly_profile(pd.DataFrame(events))
    if not monthly.empty:
        monthly = monthly.loc[monthly["facility_id"] == facility["facility_id"]]
    return fp.iloc[0], monthly


# --------------------------------------------------------------------------
# Confirmed event counts / status thresholds
# --------------------------------------------------------------------------


def test_one_confirmed_event() -> None:
    events = [_event("E1", "F1", 1)]
    result = _rt(_facility(), events)
    assert result.fingerprint["event_count"] == 1
    assert result.fingerprint["detection_count"] == 2
    assert result.fingerprint["fingerprint_observation_count"] == 1
    assert result.fingerprint["fingerprint_status"] == INSUFFICIENT_HISTORY
    assert len(result.monthly_profile) == 1
    assert result.monthly_profile[0]["month"] == 1


def test_multiple_confirmed_events() -> None:
    events = [_event(f"E{i}", "F1", month=i, detection_count=i + 1) for i in range(1, 5)]
    result = _rt(_facility(), events)
    assert result.fingerprint["event_count"] == 4
    assert result.fingerprint["detection_count"] == sum(range(2, 6))
    assert result.fingerprint["fingerprint_status"] == LIMITED_HISTORY


@pytest.mark.parametrize(
    "n,status",
    [
        (0, NO_OBSERVATIONS),
        (1, INSUFFICIENT_HISTORY),
        (2, INSUFFICIENT_HISTORY),
        (3, LIMITED_HISTORY),
        (9, LIMITED_HISTORY),
        (10, ESTABLISHED_BASELINE),
    ],
)
def test_status_thresholds(n: int, status: str) -> None:
    events = [
        _event(f"E{i}", "F1", month=((i % 12) + 1), day=(i % 27) + 1) for i in range(1, n + 1)
    ]
    result = _rt(_facility(), events)
    assert result.fingerprint["event_count"] == n
    assert result.fingerprint["fingerprint_status"] == status
    if n == 0:
        assert result.fingerprint["peak_frp_median"] is None
        assert result.monthly_profile == []


def test_day_night_mixed_handling() -> None:
    events = [
        _event("E1", "F1", 1, day_detection_count=3, night_detection_count=0),
        _event("E2", "F1", 2, day_detection_count=0, night_detection_count=3),
        _event("E3", "F1", 3, day_detection_count=2, night_detection_count=2),
    ]
    result = _rt(_facility(), events)
    fp = result.fingerprint
    assert fp["day_event_count"] == 1
    assert fp["night_event_count"] == 1
    assert fp["day_event_fraction"] == pytest.approx(1 / 3)
    assert fp["night_event_fraction"] == pytest.approx(1 / 3)
    assert fp["day_event_fraction"] + fp["night_event_fraction"] < 1.0


def test_persistence_counts() -> None:
    events = [
        _event("E1", "F1", 1, persistence_label="PERSISTENT"),
        _event("E2", "F1", 2, persistence_label="RECURRING"),
        _event("E3", "F1", 3, persistence_label="SHORT_LIVED"),
    ]
    result = _rt(_facility(), events)
    fp = result.fingerprint
    assert fp["persistent_event_count"] == 1
    assert fp["recurring_event_count"] == 1
    assert fp["short_lived_event_count"] == 1


def test_robust_stats_parity_with_batch() -> None:
    events = [
        _event("E1", "F1", 1, peak_frp=5.0, detection_count=2),
        _event("E2", "F1", 2, peak_frp=6.0, detection_count=4),
        _event("E3", "F1", 3, peak_frp=100000.0, detection_count=3),
    ]
    rt = _rt(_facility(), events)
    batch_row, _ = _batch_one(_facility(), events)
    for col in (
        "peak_frp_median",
        "peak_frp_mad",
        "peak_frp_max",
        "event_size_median",
        "duration_hours_median",
        "distance_km_median",
        "event_count",
        "detection_count",
        "fingerprint_status",
    ):
        rt_val = rt.fingerprint[col]
        batch_val = batch_row[col]
        if col == "fingerprint_status":
            assert rt_val == batch_val
        elif rt_val is None or (isinstance(batch_val, float) and pd.isna(batch_val)):
            assert rt_val is None or pd.isna(batch_val)
        else:
            assert rt_val == pytest.approx(float(batch_val))


def test_monthly_profile_parity_with_batch() -> None:
    events = [
        _event("E1", "F1", 1, detection_count=2),
        _event("E2", "F1", 1, detection_count=3),
        _event("E3", "F1", 3, detection_count=4),
    ]
    rt = _rt(_facility(), events)
    _, batch_monthly = _batch_one(_facility(), events)
    assert len(rt.monthly_profile) == len(batch_monthly) == 2
    for rt_row, (_, b_row) in zip(
        sorted(rt.monthly_profile, key=lambda r: r["month"]),
        batch_monthly.sort_values("month").iterrows(),
    ):
        assert rt_row["month"] == int(b_row["month"])
        assert rt_row["event_count"] == int(b_row["event_count"])
        assert rt_row["detection_count"] == int(b_row["detection_count"])
        assert rt_row["event_fraction"] == pytest.approx(float(b_row["event_fraction"]))


def test_ambiguous_excluded_from_primary_statistics() -> None:
    events = [
        _event(
            "E1",
            None,
            1,
            facility_association_method="AMBIGUOUS",
            facility_attribution_confidence="LOW",
            candidate_facility_ids="F1,F2",
        ),
        _event("E2", "F1", 2),
    ]
    result = _rt(_facility("F1"), events)
    assert result.fingerprint["event_count"] == 1
    assert result.fingerprint["ambiguous_candidate_opportunity_count"] == 1
    assert result.fingerprint["fingerprint_status"] == INSUFFICIENT_HISTORY


def test_no_facility_association_excluded() -> None:
    events = [
        _event(
            "E1",
            None,
            1,
            facility_association_method="NO_FACILITY_ASSOCIATION",
            facility_attribution_confidence="NONE",
            candidate_facility_ids="",
        )
    ]
    result = _rt(_facility("F1"), events)
    assert result.fingerprint["event_count"] == 0
    assert result.fingerprint["fingerprint_status"] == NO_OBSERVATIONS
    assert result.fingerprint["ambiguous_candidate_opportunity_count"] == 0
    assert result.monthly_profile == []


def test_exactly_one_facility_processed() -> None:
    # Events for F1 and F2 in the frame; adapter facilities_df has only F1.
    events = [_event("E1", "F1", 1), _event("E2", "F2", 2)]
    result = _rt(_facility("F1"), events)
    assert result.fingerprint["facility_id"] == "F1"
    assert result.fingerprint["event_count"] == 1
    assert all(r["facility_id"] == "F1" for r in result.monthly_profile)


def test_batch_realtime_outputs_equivalent() -> None:
    events = [
        _event("E1", "F1", 1, peak_frp=4.0),
        _event("E2", "F1", 2, peak_frp=8.0, day_detection_count=0, night_detection_count=2),
        _event(
            "EA",
            None,
            3,
            facility_association_method="AMBIGUOUS",
            candidate_facility_ids="F1,F9",
        ),
    ]
    rt = _rt(_facility("F1"), events)
    batch_row, batch_monthly = _batch_one(_facility("F1"), events)
    for col in batch_row.index:
        rt_val = rt.fingerprint[col]
        batch_val = batch_row[col]
        if pd.isna(batch_val):
            assert rt_val is None
        elif hasattr(batch_val, "to_pydatetime"):
            assert rt_val == batch_val.to_pydatetime()
        elif isinstance(batch_val, (int, float)) and not isinstance(batch_val, bool):
            assert rt_val == pytest.approx(float(batch_val))
        else:
            assert rt_val == batch_val
    assert len(rt.monthly_profile) == len(batch_monthly)


def test_confirmed_methods_constant() -> None:
    assert CONFIRMED_ASSOCIATION_METHODS == frozenset(
        {"WITHIN_FACILITY", "INTERSECTS_FACILITY", "NEAR_FACILITY"}
    )


def test_does_not_call_full_batch_pipeline() -> None:
    with patch("src.fingerprinting.fingerprint_pipeline.run_facility_fingerprinting") as mocked:
        _rt(_facility(), [_event("E1", "F1", 1)])
        mocked.assert_not_called()
