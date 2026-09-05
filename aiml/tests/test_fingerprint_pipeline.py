"""Integration tests for GIFT Stage I.3 (`fingerprint_pipeline`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fingerprinting.fingerprint_config import FingerprintConfig
from src.fingerprinting.fingerprint_pipeline import load_events, load_facilities, run_facility_fingerprinting, save_outputs

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
    "peak_frp": 5.0,
}


def _event(event_id: str, facility_id: str | None, month: int, day: int = 1, **overrides) -> dict:
    row = dict(_BASE_EVENT)
    row["event_id"] = event_id
    row["facility_id"] = facility_id
    row["event_start"] = f"2023-{month:02d}-{day:02d}T06:00:00+00:00"
    row["event_end"] = row["event_start"]
    row.update(overrides)
    return row


@pytest.fixture()
def synthetic_events_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _event("E1", "F_established", 1, facility_association_method="WITHIN_FACILITY", facility_attribution_confidence="HIGH", facility_distance_km=0.0),
            *[
                _event(f"E{i}", "F_established", (i % 12) + 1, day=(i % 27) + 1)
                for i in range(2, 12)
            ],
            _event("E20", "F_limited", 3, persistence_label="RECURRING"),
            _event("E21", "F_limited", 4, persistence_label="SHORT_LIVED"),
            _event("E22", "F_limited", 5, persistence_label="SHORT_LIVED"),
            _event("E30", "F_insufficient", 6),
            _event("E40", None, 7, facility_association_method="AMBIGUOUS", facility_attribution_confidence="LOW", candidate_facility_ids="F_established,F_limited"),
            _event("E41", None, 8, facility_association_method="NO_FACILITY_ASSOCIATION", facility_attribution_confidence="NONE", candidate_facility_ids=""),
        ]
    )


@pytest.fixture()
def synthetic_facilities_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"facility_id": "F_established", "facility_name": "Established Refinery", "facility_type": "REFINERY"},
            {"facility_id": "F_limited", "facility_name": "Limited Mine", "facility_type": "MINE"},
            {"facility_id": "F_insufficient", "facility_name": "Insufficient Plant", "facility_type": "POWER_PLANT"},
            {"facility_id": "F_zero", "facility_name": "Zero-observation Area", "facility_type": "INDUSTRIAL_AREA"},
        ]
    )


def test_all_facilities_represented_exactly_once(synthetic_events_df: pd.DataFrame, synthetic_facilities_df: pd.DataFrame) -> None:
    result = run_facility_fingerprinting(synthetic_events_df, synthetic_facilities_df, FingerprintConfig())
    assert len(result.fingerprints_df) == len(synthetic_facilities_df)
    assert result.fingerprints_df["facility_id"].is_unique
    assert set(result.fingerprints_df["facility_id"]) == set(synthetic_facilities_df["facility_id"])


def test_expected_statuses(synthetic_events_df: pd.DataFrame, synthetic_facilities_df: pd.DataFrame) -> None:
    result = run_facility_fingerprinting(synthetic_events_df, synthetic_facilities_df, FingerprintConfig())
    df = result.fingerprints_df.set_index("facility_id")
    assert df.loc["F_established", "fingerprint_status"] == "ESTABLISHED_BASELINE"
    assert df.loc["F_limited", "fingerprint_status"] == "LIMITED_HISTORY"
    assert df.loc["F_insufficient", "fingerprint_status"] == "INSUFFICIENT_HISTORY"
    assert df.loc["F_zero", "fingerprint_status"] == "NO_OBSERVATIONS"
    assert df.loc["F_zero", "event_count"] == 0


def test_pipeline_is_deterministic_across_repeated_runs(synthetic_events_df: pd.DataFrame, synthetic_facilities_df: pd.DataFrame) -> None:
    r1 = run_facility_fingerprinting(synthetic_events_df, synthetic_facilities_df, FingerprintConfig())
    r2 = run_facility_fingerprinting(synthetic_events_df, synthetic_facilities_df, FingerprintConfig())
    pd.testing.assert_frame_equal(r1.fingerprints_df, r2.fingerprints_df)
    pd.testing.assert_frame_equal(r1.monthly_profile_df, r2.monthly_profile_df)
    assert r1.report["fingerprint_coverage"] == r2.report["fingerprint_coverage"]


def test_report_contains_required_sections(synthetic_events_df: pd.DataFrame, synthetic_facilities_df: pd.DataFrame) -> None:
    result = run_facility_fingerprinting(synthetic_events_df, synthetic_facilities_df, FingerprintConfig())
    report = result.report
    for key in (
        "input",
        "fingerprint_coverage",
        "observation_statistics",
        "persistence_distribution_among_confirmed_events",
        "facility_type_counts_among_observed_facilities",
        "confidence_composition_among_confirmed_events",
        "temporal_coverage",
        "configuration",
        "performance",
        "limitations",
    ):
        assert key in report
    assert report["input"]["facility_count"] == len(synthetic_facilities_df)
    assert len(report["limitations"]) > 0


def test_no_literal_nan_string_and_no_negative_counts(tmp_path: Path, synthetic_events_df: pd.DataFrame, synthetic_facilities_df: pd.DataFrame) -> None:
    result = run_facility_fingerprinting(synthetic_events_df, synthetic_facilities_df, FingerprintConfig())
    fingerprints_out = tmp_path / "fp.csv"
    monthly_out = tmp_path / "mp.csv"
    save_outputs(result, fingerprints_out, monthly_out)

    reloaded = pd.read_csv(fingerprints_out)
    for col in reloaded.select_dtypes(include=["object", "str"]).columns:
        assert not (reloaded[col].astype(str) == "nan").any(), f"literal 'nan' string leaked into {col}"

    count_columns = [c for c in reloaded.columns if c.endswith("_count")]
    for col in count_columns:
        assert (reloaded[col].dropna() >= 0).all(), f"{col} has a negative value"


def test_no_source_or_anomaly_fields_anywhere(synthetic_events_df: pd.DataFrame, synthetic_facilities_df: pd.DataFrame) -> None:
    result = run_facility_fingerprinting(synthetic_events_df, synthetic_facilities_df, FingerprintConfig())
    forbidden = ("anomaly", "source_class", "industrial_fire", "wildfire", "agricultural_fire", "risk_score", "is_normal", "is_abnormal")
    all_cols = " ".join(result.fingerprints_df.columns).lower() + " " + " ".join(result.monthly_profile_df.columns).lower()
    for term in forbidden:
        assert term not in all_cols


def test_load_events_validates_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"event_id": ["E1"]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_events(path)


def test_load_events_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_events(tmp_path / "nope.csv")


def test_load_facilities_validates_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad_facilities.csv"
    pd.DataFrame({"facility_id": ["F1"]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_facilities(path)


def test_events_input_never_mutated_by_full_pipeline(synthetic_events_df: pd.DataFrame, synthetic_facilities_df: pd.DataFrame) -> None:
    original = synthetic_events_df.copy(deep=True)
    run_facility_fingerprinting(synthetic_events_df, synthetic_facilities_df, FingerprintConfig())
    pd.testing.assert_frame_equal(synthetic_events_df, original)


def test_day_night_fraction_never_exceeds_one(synthetic_events_df: pd.DataFrame, synthetic_facilities_df: pd.DataFrame) -> None:
    result = run_facility_fingerprinting(synthetic_events_df, synthetic_facilities_df, FingerprintConfig())
    df = result.fingerprints_df
    combined = (df["day_event_fraction"].fillna(0) + df["night_event_fraction"].fillna(0))
    assert (combined <= 1.0 + 1e-9).all()
