"""
Report assembly for GIFT Stage G.1 (Persistence & Recurrence Characterization).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.persistence.classification import VALID_LABELS
from src.persistence.config import PersistenceConfig


def _quantile_stats(values: pd.Series) -> dict[str, float | None]:
    values = values.dropna()
    if values.empty:
        return {"min": None, "median": None, "max": None, "mean": None}
    return {
        "min": float(values.min()),
        "median": float(values.median()),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def build_persistence_report(
    *,
    config: PersistenceConfig,
    classified_events_df: pd.DataFrame,
    input_path: str,
    processing_seconds: float,
    longest_observed_events_sample_size: int = 5,
) -> dict[str, Any]:
    """Assemble the Stage G.1 report as a JSON-serializable dict.

    Args:
        config: The `PersistenceConfig` used for this run.
        classified_events_df: Output of `classify_events`.
        input_path: Path to the Stage G `thermal_events.csv` used as
            input, for provenance.
        processing_seconds: Wall-clock seconds spent classifying.
        longest_observed_events_sample_size: How many of the
            longest-duration events to list explicitly in the report
            (a concrete, spot-checkable record that they were preserved
            as single events, not split, by this stage).

    Returns:
        A JSON-serializable dict.
    """
    n_events = len(classified_events_df)
    label_counts = classified_events_df["persistence_label"].value_counts()
    label_counts = {label: int(label_counts.get(label, 0)) for label in VALID_LABELS}

    longest = classified_events_df.sort_values("observed_duration_hours", ascending=False).head(
        longest_observed_events_sample_size
    )
    longest_sample = [
        {
            "event_id": row["event_id"],
            "detection_count": int(row["detection_count"]),
            "observed_duration_hours": float(row["observed_duration_hours"]),
            "span_days": int(row["span_days"]),
            "duty_cycle": float(row["duty_cycle"]),
            "persistence_label": row["persistence_label"],
        }
        for _, row in longest.iterrows()
    ]
    all_long_events_preserved_as_single_rows = len(longest) == len(set(longest["event_id"]))

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_stage": (
            "GIFT Stage G.1 - Persistence & Recurrence Characterization "
            "(deterministic, rule-based; built on the immutable Stage G "
            "ST-DBSCAN event table)"
        ),
        "input": {
            "path": input_path,
            "event_count": int(n_events),
        },
        "classification_config": {
            **config.to_dict(),
            "rationale": config.describe_rationale(),
            "thresholds_are_scientifically_validated": False,
        },
        "label_counts": label_counts,
        "label_percentages": (
            {label: round(count / n_events * 100.0, 4) for label, count in label_counts.items()}
            if n_events
            else {label: 0.0 for label in VALID_LABELS}
        ),
        "duty_cycle_stats": _quantile_stats(classified_events_df["duty_cycle"]),
        "span_days_stats": _quantile_stats(classified_events_df["span_days"]),
        "duration_hours_stats_by_label": {
            label: _quantile_stats(
                classified_events_df.loc[classified_events_df["persistence_label"] == label, "observed_duration_hours"]
            )
            for label in VALID_LABELS
        },
        "longest_observed_events_sample": longest_sample,
        "longest_events_preserved_as_single_rows": all_long_events_preserved_as_single_rows,
        "performance": {"processing_seconds": round(processing_seconds, 3)},
        "reproducibility": {
            "deterministic": True,
            "notes": (
                "Purely rule-based thresholding on already-computed Stage G "
                "event fields; the same input thermal_events.csv and the same "
                "PersistenceConfig values recorded above always produce "
                "identical labels."
            ),
        },
        "notes": [
            "This stage never re-runs or alters ST-DBSCAN clustering: every "
            "event_id, detection_count and detection assignment from Stage G "
            "is passed through unchanged. This stage only adds new columns.",
            "'observed_duration_hours', 'span_days' and 'duty_cycle' describe "
            "what the satellite observed, not the true physical duration of "
            "the underlying thermal source. A source may have started before "
            "its first detection, continued after its last one, or gone "
            "briefly undetected (cloud cover, sensor threshold) without "
            "actually stopping. PERSISTENT/RECURRING/SHORT_LIVED describe "
            "the *observed detection pattern* only.",
            "None of the thresholds in 'classification_config' are "
            "scientifically validated against labeled ground truth; they are "
            "documented engineering defaults intended to be tuned later.",
            f"Long-duration events (see 'longest_observed_events_sample') "
            f"remain single rows/events post-classification — this stage "
            f"cannot split or merge events, since it never touches "
            f"detection-to-event assignment.",
        ],
    }
    return _to_jsonable(report)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def save_report(report: dict[str, Any], path: str | Path) -> None:
    """Write the report dict to disk as pretty-printed JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=False)
