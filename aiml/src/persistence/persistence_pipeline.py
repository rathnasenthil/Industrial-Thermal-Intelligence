"""
End-to-end Persistence & Recurrence Characterization pipeline (GIFT Stage G.1).

Loads the Stage G `thermal_events.csv` table (treated as an immutable
baseline — never rewritten, never re-clustered), classifies each event's
persistence/recurrence pattern (`src.persistence.classification`), and
writes an augmented events table plus a JSON report.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.persistence.classification import classify_events
from src.persistence.config import PersistenceConfig
from src.persistence.persistence_report import build_persistence_report

REQUIRED_EVENT_COLUMNS: tuple[str, ...] = (
    "event_id",
    "detection_count",
    "event_start",
    "event_end",
    "observed_duration_hours",
    "distinct_detection_days",
    "max_gap_hours",
)


@dataclass
class PersistenceResult:
    """Result of running the Stage G.1 pipeline.

    Attributes:
        events_df: The Stage G events table with persistence columns
            appended (`span_days`, `duty_cycle`, `persistence_label`,
            `persistence_basis`). Same number of rows, same `event_id`
            set, as the Stage G input — this stage never splits, merges
            or drops events.
        report: JSON-serializable Stage G.1 report.
    """

    events_df: pd.DataFrame
    report: dict[str, Any]


def load_thermal_events(path: str | Path) -> pd.DataFrame:
    """Load the Stage G `thermal_events.csv` table.

    Args:
        path: Path to `thermal_events.csv` produced by
            `src.event_formation.run_event_formation`.

    Returns:
        The events DataFrame, unmodified (this function never writes to
        `path`).

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If required columns are missing.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Stage G events file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_EVENT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"'{csv_path}' is missing required column(s): {missing}. "
            "This pipeline expects the output of "
            "src.event_formation.run_event_formation (thermal_events.csv)."
        )
    return df


def run_persistence_characterization(
    events_df: pd.DataFrame, config: PersistenceConfig, input_path: str = "<in-memory>"
) -> PersistenceResult:
    """Run Stage G.1 over an already-loaded Stage G events table.

    Args:
        events_df: Output of `load_thermal_events` (or an equivalent
            DataFrame with the required columns).
        config: Classification thresholds.
        input_path: Recorded in the report for provenance only.

    Returns:
        A `PersistenceResult`. The number of rows and the set of
        `event_id` values are guaranteed identical to `events_df` — this
        stage only adds columns.
    """
    start_time = time.perf_counter()
    classified_df = classify_events(events_df, config)
    processing_seconds = time.perf_counter() - start_time

    # Invariant check: this stage must never change which/how many events
    # exist (only Stage G's ST-DBSCAN output determines that).
    assert len(classified_df) == len(events_df)
    assert set(classified_df["event_id"]) == set(events_df["event_id"])

    report = build_persistence_report(
        config=config,
        classified_events_df=classified_df,
        input_path=str(input_path),
        processing_seconds=processing_seconds,
    )
    return PersistenceResult(events_df=classified_df, report=report)


def save_events_with_persistence(events_df: pd.DataFrame, path: str | Path) -> None:
    """Write the persistence-augmented events table to CSV."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    events_df.to_csv(out_path, index=False)
