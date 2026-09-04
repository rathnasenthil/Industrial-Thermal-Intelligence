"""Integration tests for the Stage G.1 persistence-characterization pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.persistence.config import PersistenceConfig
from src.persistence.persistence_pipeline import (
    load_thermal_events,
    run_persistence_characterization,
    save_events_with_persistence,
)

_BASE_TIME = pd.Timestamp("2023-01-01T06:00:00", tz="UTC")


def _synthetic_stage_g_events() -> pd.DataFrame:
    """A small table mimicking a real thermal_events.csv, including one
    long-duration, high-duty-cycle event analogous to the real Jharia
    coal-seam-fire event found in the full Stage G run."""
    rows = [
        {
            "event_id": "EVT_0000001",
            "detection_count": 2,
            "event_start": _BASE_TIME.isoformat(),
            "event_end": (_BASE_TIME + pd.Timedelta(hours=1)).isoformat(),
            "observed_duration_hours": 1.0,
            "distinct_detection_days": 1,
            "max_gap_hours": 1.0,
            "centroid_latitude": 21.5,
            "centroid_longitude": 82.1,
            "peak_frp": 5.0,
        },
        {
            "event_id": "EVT_0000002",
            "detection_count": 4,
            "event_start": _BASE_TIME.isoformat(),
            "event_end": (_BASE_TIME + pd.Timedelta(hours=20)).isoformat(),
            "observed_duration_hours": 20.0,
            "distinct_detection_days": 1,
            "max_gap_hours": 8.0,
            "centroid_latitude": 13.0,
            "centroid_longitude": 80.2,
            "peak_frp": 3.0,
        },
        {
            "event_id": "EVT_0000003_JHARIA_LIKE",
            "detection_count": 2833,
            "event_start": _BASE_TIME.isoformat(),
            "event_end": (_BASE_TIME + pd.Timedelta(days=166)).isoformat(),
            "observed_duration_hours": 166 * 24.0,
            "distinct_detection_days": 160,
            "max_gap_hours": 40.0,
            "centroid_latitude": 23.77,
            "centroid_longitude": 86.38,
            "peak_frp": 12.0,
        },
        {
            "event_id": "EVT_0000004_RECURRING_LIKE",
            "detection_count": 5,
            "event_start": _BASE_TIME.isoformat(),
            "event_end": (_BASE_TIME + pd.Timedelta(days=40)).isoformat(),
            "observed_duration_hours": 40 * 24.0,
            "distinct_detection_days": 5,
            "max_gap_hours": 240.0,
            "centroid_latitude": 25.0,
            "centroid_longitude": 75.0,
            "peak_frp": 8.0,
        },
    ]
    return pd.DataFrame(rows)


def test_pipeline_preserves_row_count_and_event_ids() -> None:
    events_df = _synthetic_stage_g_events()
    result = run_persistence_characterization(events_df, PersistenceConfig())

    assert len(result.events_df) == len(events_df)
    assert set(result.events_df["event_id"]) == set(events_df["event_id"])


def test_pipeline_preserves_original_stage_g_columns_unchanged() -> None:
    events_df = _synthetic_stage_g_events()
    result = run_persistence_characterization(events_df, PersistenceConfig())

    for col in events_df.columns:
        pd.testing.assert_series_equal(
            result.events_df[col].reset_index(drop=True),
            events_df[col].reset_index(drop=True),
            check_names=False,
        )


def test_pipeline_adds_expected_new_columns() -> None:
    events_df = _synthetic_stage_g_events()
    result = run_persistence_characterization(events_df, PersistenceConfig())

    for col in ("span_days", "duty_cycle", "persistence_label", "persistence_basis"):
        assert col in result.events_df.columns


def test_long_duration_event_classified_persistent_and_not_split() -> None:
    events_df = _synthetic_stage_g_events()
    result = run_persistence_characterization(events_df, PersistenceConfig())

    jharia_like = result.events_df[result.events_df["event_id"] == "EVT_0000003_JHARIA_LIKE"]
    assert len(jharia_like) == 1
    assert jharia_like.iloc[0]["persistence_label"] == "PERSISTENT"
    assert jharia_like.iloc[0]["detection_count"] == 2833


def test_report_contains_expected_summary_fields() -> None:
    events_df = _synthetic_stage_g_events()
    result = run_persistence_characterization(events_df, PersistenceConfig(), input_path="synthetic.csv")

    report = result.report
    assert report["input"]["event_count"] == 4
    assert sum(report["label_counts"].values()) == 4
    assert report["classification_config"]["thresholds_are_scientifically_validated"] is False
    assert "duty_cycle_stats" in report
    assert "span_days_stats" in report
    assert len(report["longest_observed_events_sample"]) <= 4
    assert report["longest_events_preserved_as_single_rows"] is True


def test_load_thermal_events_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_thermal_events(tmp_path / "missing.csv")


def test_load_thermal_events_missing_columns_raises(tmp_path: Path) -> None:
    df = pd.DataFrame({"event_id": ["A"]})
    path = tmp_path / "bad_events.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_thermal_events(path)


def test_save_events_with_persistence_writes_csv(tmp_path: Path) -> None:
    events_df = _synthetic_stage_g_events()
    result = run_persistence_characterization(events_df, PersistenceConfig())

    out_path = tmp_path / "out" / "events_with_persistence.csv"
    save_events_with_persistence(result.events_df, out_path)

    assert out_path.exists()
    reloaded = pd.read_csv(out_path)
    assert len(reloaded) == len(events_df)
    assert "persistence_label" in reloaded.columns


def test_load_thermal_events_does_not_mutate_source_file(tmp_path: Path) -> None:
    events_df = _synthetic_stage_g_events()
    path = tmp_path / "thermal_events.csv"
    events_df.to_csv(path, index=False)
    original_bytes = path.read_bytes()

    load_thermal_events(path)

    assert path.read_bytes() == original_bytes


def test_pipeline_is_deterministic_across_runs() -> None:
    events_df = _synthetic_stage_g_events()
    result_1 = run_persistence_characterization(events_df.copy(), PersistenceConfig())
    result_2 = run_persistence_characterization(events_df.copy(), PersistenceConfig())

    pd.testing.assert_frame_equal(
        result_1.events_df.sort_values("event_id").reset_index(drop=True),
        result_2.events_df.sort_values("event_id").reset_index(drop=True),
    )
