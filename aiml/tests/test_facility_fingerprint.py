"""Tests for `src.fingerprinting.facility_fingerprint`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fingerprinting.facility_fingerprint import build_facility_fingerprints, classify_event_daynight
from src.fingerprinting.fingerprint_config import (
    ESTABLISHED_BASELINE,
    FingerprintConfig,
    INSUFFICIENT_HISTORY,
    LIMITED_HISTORY,
    NO_OBSERVATIONS,
)

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


def _facilities(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Test 1-4: fingerprint_status thresholds
# --------------------------------------------------------------------------


def test_facility_with_zero_events_is_no_observations() -> None:
    events = pd.DataFrame([_event("E1", "F_other", 1)])
    facilities = _facilities([{"facility_id": "F_zero", "facility_name": "Zero", "facility_type": "UNKNOWN"}, {"facility_id": "F_other", "facility_name": "Other", "facility_type": "MINE"}])
    result = build_facility_fingerprints(events, facilities, FingerprintConfig())
    row = result.loc[result["facility_id"] == "F_zero"].iloc[0]
    assert row["fingerprint_status"] == NO_OBSERVATIONS
    assert row["event_count"] == 0
    assert row["detection_count"] == 0
    assert pd.isna(row["peak_frp_median"])


def test_facility_with_one_event_is_insufficient_history() -> None:
    events = pd.DataFrame([_event("E1", "F1", 1)])
    facilities = _facilities([{"facility_id": "F1", "facility_name": "F1", "facility_type": "MINE"}])
    result = build_facility_fingerprints(events, facilities, FingerprintConfig())
    row = result.iloc[0]
    assert row["fingerprint_status"] == INSUFFICIENT_HISTORY
    assert row["event_count"] == 1


def test_facility_with_five_events_is_limited_history() -> None:
    events = pd.DataFrame([_event(f"E{i}", "F1", month=i) for i in range(1, 6)])
    facilities = _facilities([{"facility_id": "F1", "facility_name": "F1", "facility_type": "MINE"}])
    result = build_facility_fingerprints(events, facilities, FingerprintConfig())
    row = result.iloc[0]
    assert row["fingerprint_status"] == LIMITED_HISTORY
    assert row["event_count"] == 5


def test_facility_with_ten_events_is_established_baseline() -> None:
    events = pd.DataFrame([_event(f"E{i}", "F1", month=((i % 12) + 1), day=(i % 27) + 1) for i in range(1, 11)])
    facilities = _facilities([{"facility_id": "F1", "facility_name": "F1", "facility_type": "MINE"}])
    result = build_facility_fingerprints(events, facilities, FingerprintConfig())
    row = result.iloc[0]
    assert row["fingerprint_status"] == ESTABLISHED_BASELINE
    assert row["event_count"] == 10


# --------------------------------------------------------------------------
# Day/night classification (test 11)
# --------------------------------------------------------------------------


def test_classify_event_daynight_rules() -> None:
    result = classify_event_daynight(
        day_detection_count=pd.Series([3, 0, 2, 0]),
        night_detection_count=pd.Series([0, 4, 1, 0]),
    )
    assert list(result) == ["DAY", "NIGHT", "MIXED", "UNKNOWN"]


def test_day_night_event_counts_and_fractions() -> None:
    events = pd.DataFrame(
        [
            _event("E1", "F1", 1, day_detection_count=3, night_detection_count=0),  # DAY
            _event("E2", "F1", 2, day_detection_count=0, night_detection_count=3),  # NIGHT
            _event("E3", "F1", 3, day_detection_count=2, night_detection_count=2),  # MIXED
        ]
    )
    facilities = _facilities([{"facility_id": "F1", "facility_name": "F1", "facility_type": "MINE"}])
    result = build_facility_fingerprints(events, facilities, FingerprintConfig())
    row = result.iloc[0]
    assert row["day_event_count"] == 1
    assert row["night_event_count"] == 1
    assert row["day_event_fraction"] == pytest.approx(1 / 3)
    assert row["night_event_fraction"] == pytest.approx(1 / 3)
    # MIXED event excluded from both numerators but present in the
    # denominator -- fractions must NOT be forced to sum to 1.
    assert row["day_event_fraction"] + row["night_event_fraction"] < 1.0


# --------------------------------------------------------------------------
# Persistence fractions (test 10)
# --------------------------------------------------------------------------


def test_persistence_fractions_sum_correctly() -> None:
    events = pd.DataFrame(
        [
            _event("E1", "F1", 1, persistence_label="PERSISTENT"),
            _event("E2", "F1", 2, persistence_label="RECURRING"),
            _event("E3", "F1", 3, persistence_label="SHORT_LIVED"),
            _event("E4", "F1", 4, persistence_label="SHORT_LIVED"),
        ]
    )
    facilities = _facilities([{"facility_id": "F1", "facility_name": "F1", "facility_type": "MINE"}])
    result = build_facility_fingerprints(events, facilities, FingerprintConfig())
    row = result.iloc[0]
    assert row["persistent_event_count"] == 1
    assert row["recurring_event_count"] == 1
    assert row["short_lived_event_count"] == 2
    total_fraction = row["persistent_event_fraction"] + row["recurring_event_fraction"] + row["short_lived_event_fraction"]
    assert total_fraction == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Event count vs detection count (test 9)
# --------------------------------------------------------------------------


def test_event_count_differs_from_detection_count() -> None:
    events = pd.DataFrame(
        [
            _event("E1", "F1", 1, detection_count=50),
            _event("E2", "F1", 2, detection_count=3),
        ]
    )
    facilities = _facilities([{"facility_id": "F1", "facility_name": "F1", "facility_type": "MINE"}])
    result = build_facility_fingerprints(events, facilities, FingerprintConfig())
    row = result.iloc[0]
    assert row["event_count"] == 2
    assert row["detection_count"] == 53
    assert row["event_count"] != row["detection_count"]


# --------------------------------------------------------------------------
# Missing FRP (test 13/14)
# --------------------------------------------------------------------------


def test_missing_frp_does_not_become_zero() -> None:
    events = pd.DataFrame(
        [
            _event("E1", "F1", 1, peak_frp=np.nan),
            _event("E2", "F1", 2, peak_frp=np.nan),
        ]
    )
    facilities = _facilities([{"facility_id": "F1", "facility_name": "F1", "facility_type": "MINE"}])
    result = build_facility_fingerprints(events, facilities, FingerprintConfig())
    row = result.iloc[0]
    assert pd.isna(row["peak_frp_median"])
    assert pd.isna(row["peak_frp_max"])
    assert row["event_count"] == 2  # events themselves are still counted


def test_extreme_frp_does_not_distort_median() -> None:
    events = pd.DataFrame(
        [_event("E1", "F1", 1, peak_frp=5.0), _event("E2", "F1", 2, peak_frp=6.0), _event("E3", "F1", 3, peak_frp=100000.0)]
    )
    facilities = _facilities([{"facility_id": "F1", "facility_name": "F1", "facility_type": "MINE"}])
    result = build_facility_fingerprints(events, facilities, FingerprintConfig())
    row = result.iloc[0]
    assert row["peak_frp_median"] == 6.0
    assert row["peak_frp_max"] == 100000.0  # preserved, never discarded


# --------------------------------------------------------------------------
# Ambiguous events never counted as confirmed observations (test 16)
# --------------------------------------------------------------------------


def test_ambiguous_events_are_not_confirmed_observations() -> None:
    events = pd.DataFrame(
        [
            _event("E1", None, 1, facility_association_method="AMBIGUOUS", facility_attribution_confidence="LOW", candidate_facility_ids="F1,F2"),
            _event("E2", None, 2, facility_association_method="NO_FACILITY_ASSOCIATION", facility_attribution_confidence="NONE", candidate_facility_ids=""),
        ]
    )
    facilities = _facilities(
        [
            {"facility_id": "F1", "facility_name": "F1", "facility_type": "MINE"},
            {"facility_id": "F2", "facility_name": "F2", "facility_type": "MINE"},
        ]
    )
    result = build_facility_fingerprints(events, facilities, FingerprintConfig())
    for facility_id in ("F1", "F2"):
        row = result.loc[result["facility_id"] == facility_id].iloc[0]
        assert row["event_count"] == 0
        assert row["fingerprint_status"] == NO_OBSERVATIONS
        # Recorded informationally, but never inflates event_count.
        assert row["ambiguous_candidate_opportunity_count"] == 1


# --------------------------------------------------------------------------
# Every facility remains represented (test 15)
# --------------------------------------------------------------------------


def test_every_facility_remains_represented_and_unique() -> None:
    events = pd.DataFrame([_event("E1", "F1", 1)])
    facilities = _facilities(
        [
            {"facility_id": "F1", "facility_name": "F1", "facility_type": "MINE"},
            {"facility_id": "F2", "facility_name": "F2", "facility_type": "REFINERY"},
            {"facility_id": "F3", "facility_name": "F3", "facility_type": "UNKNOWN"},
        ]
    )
    result = build_facility_fingerprints(events, facilities, FingerprintConfig())
    assert len(result) == 3
    assert result["facility_id"].is_unique
    assert set(result["facility_id"]) == {"F1", "F2", "F3"}


# --------------------------------------------------------------------------
# Existing event rows are never modified (test 17)
# --------------------------------------------------------------------------


def test_events_dataframe_is_not_mutated() -> None:
    events = pd.DataFrame([_event("E1", "F1", 1)])
    original = events.copy(deep=True)
    facilities = _facilities([{"facility_id": "F1", "facility_name": "F1", "facility_type": "MINE"}])
    build_facility_fingerprints(events, facilities, FingerprintConfig())
    pd.testing.assert_frame_equal(events, original)


# --------------------------------------------------------------------------
# Output ordering deterministic (test 18)
# --------------------------------------------------------------------------


def test_output_is_sorted_by_facility_id() -> None:
    events = pd.DataFrame([_event("E1", "F_z", 1), _event("E2", "F_a", 2)])
    facilities = _facilities(
        [
            {"facility_id": "F_z", "facility_name": "Z", "facility_type": "MINE"},
            {"facility_id": "F_a", "facility_name": "A", "facility_type": "MINE"},
            {"facility_id": "F_m", "facility_name": "M", "facility_type": "MINE"},
        ]
    )
    result = build_facility_fingerprints(events, facilities, FingerprintConfig())
    assert list(result["facility_id"]) == ["F_a", "F_m", "F_z"]


def test_missing_required_columns_raise() -> None:
    with pytest.raises(ValueError):
        build_facility_fingerprints(pd.DataFrame({"event_id": ["E1"]}), _facilities([{"facility_id": "F1", "facility_name": "F1", "facility_type": "MINE"}]), FingerprintConfig())
    with pytest.raises(ValueError):
        build_facility_fingerprints(pd.DataFrame([_event("E1", "F1", 1)]), pd.DataFrame({"facility_id": ["F1"]}), FingerprintConfig())


def test_within_intersects_near_counts() -> None:
    events = pd.DataFrame(
        [
            _event("E1", "F1", 1, facility_association_method="WITHIN_FACILITY", facility_distance_km=0.0),
            _event("E2", "F1", 2, facility_association_method="INTERSECTS_FACILITY", facility_distance_km=0.1),
            _event("E3", "F1", 3, facility_association_method="NEAR_FACILITY", facility_distance_km=2.0),
        ]
    )
    facilities = _facilities([{"facility_id": "F1", "facility_name": "F1", "facility_type": "MINE"}])
    result = build_facility_fingerprints(events, facilities, FingerprintConfig())
    row = result.iloc[0]
    assert row["within_facility_count"] == 1
    assert row["intersects_facility_count"] == 1
    assert row["near_facility_count"] == 1


def test_confidence_counts() -> None:
    events = pd.DataFrame(
        [
            _event("E1", "F1", 1, facility_attribution_confidence="HIGH"),
            _event("E2", "F1", 2, facility_attribution_confidence="MEDIUM"),
            _event("E3", "F1", 3, facility_attribution_confidence="MEDIUM"),
            _event("E4", "F1", 4, facility_attribution_confidence="LOW"),
        ]
    )
    facilities = _facilities([{"facility_id": "F1", "facility_name": "F1", "facility_type": "MINE"}])
    result = build_facility_fingerprints(events, facilities, FingerprintConfig())
    row = result.iloc[0]
    assert row["high_confidence_event_count"] == 1
    assert row["medium_confidence_event_count"] == 2
    assert row["low_confidence_event_count"] == 1


def test_no_source_classification_fields_in_output() -> None:
    events = pd.DataFrame([_event("E1", "F1", 1)])
    facilities = _facilities([{"facility_id": "F1", "facility_name": "F1", "facility_type": "MINE"}])
    result = build_facility_fingerprints(events, facilities, FingerprintConfig())
    forbidden = ("anomaly", "source_class", "industrial_fire", "wildfire", "risk_score")
    columns_lower = " ".join(result.columns).lower()
    for term in forbidden:
        assert term not in columns_lower
