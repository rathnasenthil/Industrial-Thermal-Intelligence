"""
Incremental Stage G.1 persistence characterization (AIML realtime adapter).

Persistence means: thermal activity is *repeatedly observed over time*.
It does **not** mean confirmed fire, industrial source, or dangerous event.
This module never performs fire classification.

Why realtime G.1 cannot replay the full batch pipeline
------------------------------------------------------
Batch ``run_persistence_characterization()`` classifies every row in a Stage G
``thermal_events`` table (historically ~180k events). On each NRT poll only a
handful of events receive new detections. Re-running the batch entrypoint would
re-touch unrelated historical rows and is unnecessary.

Instead: for the *one* affected event, recompute Stage G temporal features from
that event's stored detections, then call the same ``classify_events`` /
``compute_span_days`` / ``compute_duty_cycle`` formulas used by batch G.1.

Why batch formulas are reused
-----------------------------
Existing ``src.persistence`` is the semantic source of truth. Inventing parallel
thresholds would drift from Stage VII / offline validation. Realtime only adapts
*scope* (one event), not *rules*.

Why recompute from the detection set
------------------------------------
Deriving features from the full detection list (via ``event_detections`` →
FIRMS timestamps) is idempotent: processing the same event twice without a new
detection yields identical values. Fragile incremental counters are avoided.

Single-observation realtime events
----------------------------------
Phase 3 may create events with one observation (batch ST-DBSCAN ``min_samples=2``
would mark those as noise). Under default ``PersistenceConfig``,
``detection_count < 3`` → ``INSUFFICIENT_OBSERVATIONS``. That is correct: one
overpass is not a persistence pattern.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import isnan
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from src.persistence.classification import classify_events
from src.persistence.config import DEFAULT_CONFIG, PersistenceConfig

# Instrumentation: realtime path must never call the full-batch orchestrator.
# Tests assert this remains zero (or that the batch entrypoint is not imported
# for execution here).
_BATCH_PIPELINE_INVOCATIONS = 0


@dataclass(frozen=True)
class PersistenceFeatures:
    """Framework-independent G.1 result for one thermal event."""

    event_id: str
    detection_count: int
    distinct_detection_days: int
    span_days: int
    observed_duration_hours: float
    duty_cycle: float
    mean_gap_hours: Optional[float]
    max_gap_hours: Optional[float]
    persistence_label: str
    persistence_basis: str
    event_start: datetime
    event_end: datetime

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["event_start"] = self.event_start.isoformat()
        out["event_end"] = self.event_end.isoformat()
        return out


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _nan_to_none(value: float) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if isnan(f):
        return None
    return f


def compute_temporal_features_from_datetimes(
    datetimes: Sequence[datetime],
) -> dict[str, Any]:
    """
    Stage G temporal fields for one event, matching ``compute_event_row``.

    Formulas (from ``src.event_formation.event_features.compute_event_row``):
    - event_start / event_end = min / max of sorted acquisition times
    - observed_duration_hours = (end − start) in hours
    - distinct_detection_days = nunique of calendar dates
    - mean_gap_hours / max_gap_hours = from consecutive sorted diffs;
      NaN when only one detection
    - detection_count = number of valid timestamps

    Null / invalid timestamps are dropped. If none remain, raises ValueError.
    """
    valid: list[datetime] = []
    for raw in datetimes:
        if raw is None:
            continue
        if not isinstance(raw, datetime):
            continue
        valid.append(_ensure_utc(raw))

    if not valid:
        raise ValueError(
            "no valid detection timestamps for persistence characterization"
        )

    series = pd.Series(valid, dtype="datetime64[ns, UTC]").sort_values()
    event_start = series.iloc[0].to_pydatetime()
    event_end = series.iloc[-1].to_pydatetime()
    observed_duration_hours = (event_end - event_start).total_seconds() / 3600.0
    distinct_days = int(series.dt.date.nunique())

    if len(series) > 1:
        gaps_hours = series.diff().dropna().dt.total_seconds() / 3600.0
        mean_gap_hours = float(gaps_hours.mean())
        max_gap_hours = float(gaps_hours.max())
    else:
        mean_gap_hours = float("nan")
        max_gap_hours = float("nan")

    return {
        "detection_count": len(series),
        "event_start": event_start,
        "event_end": event_end,
        "observed_duration_hours": float(observed_duration_hours),
        "distinct_detection_days": distinct_days,
        "mean_gap_hours": mean_gap_hours,
        "max_gap_hours": max_gap_hours,
    }


def process_event_persistence(
    event_id: str,
    detection_datetimes: Sequence[datetime],
    *,
    config: Optional[PersistenceConfig] = None,
) -> PersistenceFeatures:
    """
    Recompute G.1 persistence for **one** thermal event from its detections.

    Does not call ``run_persistence_characterization()`` over a full events
    table. Uses ``classify_events`` with the same ``PersistenceConfig`` defaults
    as batch Stage G.1.

    Args:
        event_id: Stable thermal event identifier (e.g. ``EVT_#######``).
        detection_datetimes: Acquisition times of observations linked to this
            event (order does not matter; nulls ignored).
        config: Optional override; defaults to ``DEFAULT_CONFIG``.

    Returns:
        PersistenceFeatures with batch-identical classification semantics.
    """
    cfg = config or DEFAULT_CONFIG
    temporal = compute_temporal_features_from_datetimes(detection_datetimes)

    # One-row Stage G–compatible frame for classify_events.
    events_df = pd.DataFrame(
        [
            {
                "event_id": event_id,
                "detection_count": temporal["detection_count"],
                "event_start": temporal["event_start"].isoformat(),
                "event_end": temporal["event_end"].isoformat(),
                "observed_duration_hours": temporal["observed_duration_hours"],
                "distinct_detection_days": temporal["distinct_detection_days"],
                "max_gap_hours": temporal["max_gap_hours"],
            }
        ]
    )
    classified = classify_events(events_df, cfg)
    row = classified.iloc[0]

    return PersistenceFeatures(
        event_id=event_id,
        detection_count=int(row["detection_count"]),
        distinct_detection_days=int(row["distinct_detection_days"]),
        span_days=int(row["span_days"]),
        observed_duration_hours=float(row["observed_duration_hours"]),
        duty_cycle=float(row["duty_cycle"]),
        mean_gap_hours=_nan_to_none(temporal["mean_gap_hours"]),
        max_gap_hours=_nan_to_none(float(row["max_gap_hours"])),
        persistence_label=str(row["persistence_label"]),
        persistence_basis=str(row["persistence_basis"]),
        event_start=temporal["event_start"],
        event_end=temporal["event_end"],
    )


def process_event_persistence_from_mapping(
    event: Mapping[str, Any],
    detections: Sequence[Mapping[str, Any] | datetime],
    *,
    config: Optional[PersistenceConfig] = None,
    datetime_key: str = "acq_datetime",
) -> PersistenceFeatures:
    """
    Convenience wrapper: ``process_event_persistence(event, detections)``.

    ``event`` must provide ``event_id``. Each detection is either a datetime or
    a mapping with ``datetime_key`` (default ``acq_datetime``).
    """
    event_id = str(event["event_id"])
    times: list[datetime] = []
    for item in detections:
        if isinstance(item, datetime):
            times.append(item)
        elif isinstance(item, Mapping):
            times.append(item.get(datetime_key))  # type: ignore[arg-type]
        else:
            times.append(item)  # type: ignore[arg-type]
    return process_event_persistence(event_id, times, config=config)


def batch_pipeline_invocation_count() -> int:
    """Test hook: must stay 0 on the realtime path."""
    return _BATCH_PIPELINE_INVOCATIONS
