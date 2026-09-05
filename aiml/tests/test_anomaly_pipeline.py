"""Integration tests for GIFT Stage I.4 anomaly detection pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection.anomaly_pipeline import run_anomaly_detection, save_outputs
from src.anomaly_detection.config import AnomalyConfig, INSUFFICIENT_HISTORY, NORMAL


def _event(
    event_id: str,
    facility_id: str | None,
    day: int,
    *,
    method: str = "NEAR_FACILITY",
    peak_frp: float = 5.0,
    detection_count: int = 2,
    duration: float = 1.0,
    distance: float | None = 1.0,
    persistence: str = "SHORT_LIVED",
    month: int = 1,
) -> dict:
    return {
        "event_id": event_id,
        "facility_id": facility_id,
        "facility_name": "Test" if facility_id else None,
        "facility_type": "MINE" if facility_id else None,
        "facility_association_method": method,
        "facility_attribution_confidence": "MEDIUM" if facility_id else "NONE",
        "facility_distance_km": distance,
        "candidate_facility_count": 1 if facility_id else 0,
        "candidate_facility_ids": facility_id or "",
        "event_start": f"2023-{month:02d}-{day:02d}T06:00:00+00:00",
        "event_end": f"2023-{month:02d}-{day:02d}T07:00:00+00:00",
        "peak_frp": peak_frp,
        "detection_count": detection_count,
        "observed_duration_hours": duration,
        "persistence_label": persistence,
        "persistence_basis": "test",
    }


@pytest.fixture()
def synthetic_events() -> pd.DataFrame:
    rows = []
    # Facility F1: 12 events — enough for established baseline walk-forward
    for i in range(1, 13):
        rows.append(
            _event(
                f"F1_E{i:02d}",
                "F1",
                day=min(i, 28),
                month=((i - 1) % 12) + 1,
                peak_frp=5.0 if i < 12 else 80.0,  # last event anomalous FRP
                detection_count=2 if i < 12 else 40,
            )
        )
    # Facility F2: only 1 event → insufficient
    rows.append(_event("F2_E01", "F2", day=1, peak_frp=9.0))
    # Ambiguous — must not enter baseline
    rows.append(
        _event(
            "AMB_01",
            None,
            day=15,
            method="AMBIGUOUS",
            distance=None,
            peak_frp=99.0,
        )
    )
    # No association
    rows.append(
        _event(
            "NOA_01",
            None,
            day=16,
            method="NO_FACILITY_ASSOCIATION",
            distance=None,
            peak_frp=1.0,
        )
    )
    return pd.DataFrame(rows)


def test_row_count_preserved(synthetic_events: pd.DataFrame) -> None:
    result = run_anomaly_detection(synthetic_events, AnomalyConfig())
    assert len(result.events_df) == len(synthetic_events)
    assert result.events_df["event_id"].is_unique


def test_no_facility_association_preserved(synthetic_events: pd.DataFrame) -> None:
    result = run_anomaly_detection(synthetic_events, AnomalyConfig())
    row = result.events_df.loc[result.events_df["event_id"] == "NOA_01"].iloc[0]
    assert row["anomaly_status"] == INSUFFICIENT_HISTORY
    assert row["anomaly_unavailable_reason"] == "NO_FACILITY_ASSOCIATION"
    assert pd.isna(row["anomaly_score"])


def test_ambiguous_not_assigned_and_insufficient(synthetic_events: pd.DataFrame) -> None:
    result = run_anomaly_detection(synthetic_events, AnomalyConfig())
    row = result.events_df.loc[result.events_df["event_id"] == "AMB_01"].iloc[0]
    assert row["anomaly_status"] == INSUFFICIENT_HISTORY
    assert row["anomaly_unavailable_reason"] == "AMBIGUOUS_ASSOCIATION"
    assert pd.isna(row["facility_id"]) or row["facility_id"] is None or str(row["facility_id"]) == "nan"


def test_ambiguous_does_not_contaminate_facility_baseline(synthetic_events: pd.DataFrame) -> None:
    # Ambiguous has peak_frp=99; if it contaminated F1, later F1 medians would shift.
    # F1 events only — last event baseline should be median of first 11 F1 events (~5).
    result = run_anomaly_detection(synthetic_events, AnomalyConfig())
    last = result.events_df.loc[result.events_df["event_id"] == "F1_E12"].iloc[0]
    assert last["baseline_observation_count"] == 11
    assert last["baseline_peak_frp_median"] == pytest.approx(5.0)


def test_first_facility_events_insufficient(synthetic_events: pd.DataFrame) -> None:
    result = run_anomaly_detection(synthetic_events, AnomalyConfig())
    for eid in ("F1_E01", "F1_E02", "F2_E01"):
        row = result.events_df.loc[result.events_df["event_id"] == eid].iloc[0]
        assert row["anomaly_status"] == INSUFFICIENT_HISTORY
        assert pd.isna(row["anomaly_score"])


def test_later_event_can_be_scored(synthetic_events: pd.DataFrame) -> None:
    result = run_anomaly_detection(synthetic_events, AnomalyConfig())
    last = result.events_df.loc[result.events_df["event_id"] == "F1_E12"].iloc[0]
    assert last["anomaly_status"] in ("ELEVATED", "ANOMALOUS", NORMAL)
    assert last["anomaly_score"] is not None and last["anomaly_score"] > 0
    assert last["peak_frp_deviation"] is not None


def test_missing_feature_stays_null_not_zero() -> None:
    # Event with missing peak_frp among established history.
    rows = [_event(f"E{i}", "F1", day=i, peak_frp=5.0) for i in range(1, 11)]
    rows.append(_event("E11", "F1", day=11, peak_frp=float("nan")))
    events = pd.DataFrame(rows)
    result = run_anomaly_detection(events, AnomalyConfig())
    row = result.events_df.loc[result.events_df["event_id"] == "E11"].iloc[0]
    assert pd.isna(row["peak_frp_deviation"])


def test_events_dataframe_not_mutated(synthetic_events: pd.DataFrame) -> None:
    original = synthetic_events.copy(deep=True)
    run_anomaly_detection(synthetic_events, AnomalyConfig())
    pd.testing.assert_frame_equal(synthetic_events, original)


def test_pipeline_deterministic(synthetic_events: pd.DataFrame) -> None:
    r1 = run_anomaly_detection(synthetic_events, AnomalyConfig())
    r2 = run_anomaly_detection(synthetic_events, AnomalyConfig())
    cols = [
        "event_id",
        "anomaly_score",
        "anomaly_status",
        "anomaly_confidence",
        "peak_frp_deviation",
        "baseline_observation_count",
        "anomaly_explanation",
    ]
    pd.testing.assert_frame_equal(r1.events_df[cols], r2.events_df[cols])


def test_shuffled_input_same_output(synthetic_events: pd.DataFrame) -> None:
    shuffled = synthetic_events.sample(frac=1.0, random_state=42).reset_index(drop=True)
    r1 = run_anomaly_detection(synthetic_events, AnomalyConfig())
    r2 = run_anomaly_detection(shuffled, AnomalyConfig())
    m1 = r1.events_df.set_index("event_id").sort_index()
    m2 = r2.events_df.set_index("event_id").sort_index()
    for col in ("anomaly_score", "anomaly_status", "baseline_observation_count", "peak_frp_deviation"):
        pd.testing.assert_series_equal(m1[col], m2[col], check_names=False)


def test_no_negative_scores_no_nan_string(tmp_path: Path, synthetic_events: pd.DataFrame) -> None:
    result = run_anomaly_detection(synthetic_events, AnomalyConfig())
    out = tmp_path / "out.csv"
    save_outputs(result, out)
    reloaded = pd.read_csv(out)
    scores = reloaded["anomaly_score"].dropna()
    assert (scores >= 0).all()
    for col in reloaded.select_dtypes(include=["object", "str"]).columns:
        # Literal string "nan" should not appear (pandas NaN is fine as missing).
        assert not ((reloaded[col] == "nan") & reloaded[col].notna()).any()


def test_no_source_classification_fields(synthetic_events: pd.DataFrame) -> None:
    result = run_anomaly_detection(synthetic_events, AnomalyConfig())
    forbidden = ("industrial_fire", "wildfire", "agricultural", "risk_score", "source_class")
    cols = " ".join(result.events_df.columns).lower()
    for term in forbidden:
        assert term not in cols


def test_persistence_label_consumed_not_recomputed(synthetic_events: pd.DataFrame) -> None:
    # Pipeline must preserve original persistence_label column unchanged.
    result = run_anomaly_detection(synthetic_events, AnomalyConfig())
    merged = synthetic_events[["event_id", "persistence_label"]].merge(
        result.events_df[["event_id", "persistence_label"]],
        on="event_id",
        suffixes=("_in", "_out"),
    )
    assert (merged["persistence_label_in"] == merged["persistence_label_out"]).all()


def test_report_sections(synthetic_events: pd.DataFrame) -> None:
    result = run_anomaly_detection(synthetic_events, AnomalyConfig())
    for key in (
        "input",
        "history_at_scoring_time",
        "anomaly_status_counts",
        "anomaly_confidence_counts",
        "feature_availability",
        "configuration",
        "leakage_validation",
        "limitations",
    ):
        assert key in result.report
    assert result.report["leakage_validation"]["walk_forward_prior_only"] is True
    assert result.report["input"]["total_events_processed"] == len(synthetic_events)
