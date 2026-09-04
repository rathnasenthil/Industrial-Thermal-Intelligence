"""Integration tests for the full Stage G event-formation pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from src.event_formation.config import STDBSCANConfig
from src.event_formation.event_pipeline import run_event_formation

_BASE_TIME = pd.Timestamp("2023-01-01T06:00:00", tz="UTC")


def _make_detections() -> pd.DataFrame:
    rows = []

    # Cluster A: 3 detections close together in space and time.
    for i in range(3):
        rows.append(
            {
                "latitude": 21.500 + i * 0.001,
                "longitude": 82.100 + i * 0.001,
                "acq_datetime": _BASE_TIME + pd.Timedelta(hours=i * 2),
                "frp": 5.0 + i,
                "frp_valid": True,
                "bright_ti4": 330.0 + i,
                "bright_ti5": 290.0 + i,
                "confidence": "n",
                "daynight": "D",
            }
        )

    # Cluster B: 2 detections, far away, one has missing FRP.
    for i in range(2):
        rows.append(
            {
                "latitude": 13.000 + i * 0.001,
                "longitude": 80.200 + i * 0.001,
                "acq_datetime": _BASE_TIME + pd.Timedelta(hours=i * 3),
                "frp": None if i == 0 else 8.0,
                "frp_valid": False if i == 0 else True,
                "bright_ti4": 320.0,
                "bright_ti5": 285.0,
                "confidence": "l",
                "daynight": "N",
            }
        )

    # One isolated noise detection.
    rows.append(
        {
            "latitude": 5.000,
            "longitude": 60.000,
            "acq_datetime": _BASE_TIME,
            "frp": 1.0,
            "frp_valid": True,
            "bright_ti4": 310.0,
            "bright_ti5": 280.0,
            "confidence": "h",
            "daynight": "D",
        }
    )

    return pd.DataFrame(rows)


@pytest.fixture()
def config() -> STDBSCANConfig:
    return STDBSCANConfig(spatial_eps_km=1.5, temporal_eps_hours=36, min_samples=2)


def test_pipeline_produces_expected_event_and_noise_counts(config: STDBSCANConfig) -> None:
    detections = _make_detections()

    result = run_event_formation(detections, config, measure_memory=False)

    assert len(result.events_df) == 2
    assert len(result.noise_df) == 1
    assert len(result.detections_df) == 5  # 3 + 2 clustered detections


def test_pipeline_links_detections_to_events_via_event_id(config: STDBSCANConfig) -> None:
    detections = _make_detections()
    result = run_event_formation(detections, config, measure_memory=False)

    event_ids_in_events_table = set(result.events_df["event_id"])
    event_ids_in_detections_table = set(result.detections_df["event_id"])
    assert event_ids_in_events_table == event_ids_in_detections_table

    for event_id in event_ids_in_events_table:
        expected_count = result.events_df.loc[result.events_df["event_id"] == event_id, "detection_count"].iloc[0]
        actual_count = (result.detections_df["event_id"] == event_id).sum()
        assert expected_count == actual_count


def test_noise_detections_are_preserved_not_deleted(config: STDBSCANConfig) -> None:
    detections = _make_detections()
    result = run_event_formation(detections, config, measure_memory=False)

    assert result.noise_df.iloc[0]["latitude"] == pytest.approx(5.000)
    assert "noise_reason" in result.noise_df.columns
    assert "event_id" not in result.noise_df.columns


def test_report_contains_required_summary_fields(config: STDBSCANConfig) -> None:
    detections = _make_detections()
    result = run_event_formation(detections, config, measure_memory=False, input_path="synthetic_test.csv")

    report = result.report
    assert report["input"]["detection_count"] == 6
    assert report["counts"]["event_count"] == 2
    assert report["counts"]["noise_detection_count"] == 1
    assert report["counts"]["clustered_detection_count"] == 5
    assert report["clustering_config"]["spatial_eps_km"] == 1.5
    assert report["clustering_config"]["parameters_are_scientifically_validated"] is False
    assert "event_size_stats" in report
    assert "event_duration_hours_stats" in report
    assert "event_peak_frp_stats" in report
    assert report["performance"]["processing_seconds"] >= 0


def test_pipeline_is_deterministic_across_runs(config: STDBSCANConfig) -> None:
    detections = _make_detections()

    result_1 = run_event_formation(detections.copy(), config, measure_memory=False)
    result_2 = run_event_formation(detections.copy(), config, measure_memory=False)

    pd.testing.assert_frame_equal(
        result_1.events_df.drop(columns=["event_id"]), result_2.events_df.drop(columns=["event_id"])
    )
    assert len(result_1.detections_df) == len(result_2.detections_df)
    assert len(result_1.noise_df) == len(result_2.noise_df)


def test_missing_frp_in_cluster_does_not_fabricate_value(config: STDBSCANConfig) -> None:
    detections = _make_detections()
    result = run_event_formation(detections, config, measure_memory=False)

    cluster_b_row = result.events_df[result.events_df["detection_count"] == 2].iloc[0]
    # Cluster B has one missing FRP and one valid (8.0) -> stats computed
    # only from the valid one, not fabricated as 0.
    assert cluster_b_row["frp_valid_count"] == 1
    assert cluster_b_row["peak_frp"] == pytest.approx(8.0)
