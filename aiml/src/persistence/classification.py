"""
Deterministic persistence/recurrence classification for GIFT Stage G.1.

Given the event-level table produced by GIFT Stage G (one row per
ST-DBSCAN cluster, with `detection_count`, `event_start`, `event_end`,
`observed_duration_hours`, `distinct_detection_days`, `max_gap_hours`
already computed), this module adds two new *deterministic, rule-based*
metrics — `span_days` and `duty_cycle` — and assigns each event one of
four labels:

* ``INSUFFICIENT_OBSERVATIONS`` — too few detections to say anything
  about a temporal pattern beyond "it happened".
* ``SHORT_LIVED`` — a brief episode (small observed span).
* ``PERSISTENT`` — activity observed on a large share of the days it
  spans, without long silent gaps.
* ``RECURRING`` — activity observed repeatedly over an extended span, but
  not densely/continuously enough to call PERSISTENT.

IMPORTANT — this is purely a re-description of the *observed* detection
pattern already produced by Stage G. It does NOT re-cluster anything, and
it does NOT determine the true physical start/end/duration of whatever
produced the thermal signal: FIRMS only records discrete satellite
overpasses, so the real thermal source may well have started before the
first detection and/or continued after the last one, and gaps between
detections do not prove the source was inactive (e.g. cloud cover, or a
sub-detection-threshold signal). `observed_duration_hours` (from Stage G)
and `span_days`/`duty_cycle` (added here) all describe what was
*observed*, not the physical ground truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.persistence.config import PersistenceConfig

INSUFFICIENT_OBSERVATIONS = "INSUFFICIENT_OBSERVATIONS"
SHORT_LIVED = "SHORT_LIVED"
PERSISTENT = "PERSISTENT"
RECURRING = "RECURRING"

VALID_LABELS: tuple[str, ...] = (INSUFFICIENT_OBSERVATIONS, SHORT_LIVED, PERSISTENT, RECURRING)


def compute_span_days(event_start: pd.Series, event_end: pd.Series) -> pd.Series:
    """Calendar-day span of an event (inclusive of both start and end days).

    This is deliberately based on calendar dates (not raw hours / 24) so
    that e.g. an event observed at 23:50 on day 1 and 00:10 on day 2 —
    only 20 minutes of `observed_duration_hours` — correctly counts as
    spanning 2 distinct calendar days for duty-cycle purposes, matching
    how `distinct_detection_days` (from Stage G) is computed.

    Args:
        event_start: Event start timestamps (timezone-aware UTC).
        event_end: Event end timestamps (timezone-aware UTC).

    Returns:
        Integer Series: number of calendar days spanned (always >= 1).
    """
    return ((event_end.dt.normalize() - event_start.dt.normalize()).dt.days + 1).astype(int)


def compute_duty_cycle(distinct_detection_days: pd.Series, span_days: pd.Series) -> pd.Series:
    """Fraction of an event's spanned calendar days that had a detection.

    Args:
        distinct_detection_days: Number of distinct calendar days with at
            least one detection (from Stage G).
        span_days: Calendar-day span of the event (see
            `compute_span_days`); always >= 1, so no division-by-zero
            handling is needed.

    Returns:
        Float Series in (0, 1]. A value of 1.0 means the event was
        detected on every single day within its observed span; lower
        values mean detections were sparser relative to the span.
    """
    return (distinct_detection_days / span_days).clip(upper=1.0)


def classify_events(events_df: pd.DataFrame, config: PersistenceConfig) -> pd.DataFrame:
    """Classify each event's persistence/recurrence pattern.

    Args:
        events_df: The Stage G `thermal_events` table (or a compatible
            DataFrame) containing at least: `detection_count`,
            `event_start`, `event_end`, `observed_duration_hours`,
            `distinct_detection_days`, `max_gap_hours`.
        config: Classification thresholds.

    Returns:
        A copy of `events_df` with four new columns appended:
        `span_days`, `duty_cycle`, `persistence_label` and
        `persistence_basis` (a short human-readable explanation of which
        rule fired).

    Raises:
        ValueError: If required columns are missing.
    """
    required = {
        "detection_count",
        "event_start",
        "event_end",
        "observed_duration_hours",
        "distinct_detection_days",
        "max_gap_hours",
    }
    missing = required - set(events_df.columns)
    if missing:
        raise ValueError(f"events_df is missing required column(s): {sorted(missing)}")

    out = events_df.copy()
    event_start = pd.to_datetime(out["event_start"], utc=True)
    event_end = pd.to_datetime(out["event_end"], utc=True)

    out["span_days"] = compute_span_days(event_start, event_end)
    out["duty_cycle"] = compute_duty_cycle(out["distinct_detection_days"], out["span_days"])

    insufficient = out["detection_count"] < config.min_detections_for_classification
    short_lived = (~insufficient) & (out["observed_duration_hours"] <= config.short_lived_max_duration_hours)
    # PERSISTENT uses OR (not AND) between the two qualifying conditions:
    # either a high duty cycle or the absence of any long internal gap is
    # sufficient on its own. This is deliberate — see PersistenceConfig's
    # docstring for why an AND rule would wrongly downgrade genuinely
    # persistent, high-duty-cycle sources that happen to have exactly one
    # longer pause (e.g. one missed overpass pair due to cloud cover).
    high_duty_cycle = out["duty_cycle"] >= config.persistent_min_duty_cycle
    no_long_gap = out["max_gap_hours"] <= config.persistent_max_gap_hours
    persistent = (~insufficient) & (~short_lived) & (high_duty_cycle | no_long_gap)
    recurring = (~insufficient) & (~short_lived) & (~persistent)

    labels = np.select(
        [insufficient, short_lived, persistent, recurring],
        [INSUFFICIENT_OBSERVATIONS, SHORT_LIVED, PERSISTENT, RECURRING],
        default=INSUFFICIENT_OBSERVATIONS,
    )
    out["persistence_label"] = labels

    out["persistence_basis"] = np.select(
        [insufficient, short_lived, persistent, recurring],
        [
            (
                f"detection_count < {config.min_detections_for_classification} "
                "(too few observations to characterize a temporal pattern)."
            ),
            (
                f"observed_duration_hours <= {config.short_lived_max_duration_hours} "
                "(brief observed episode)."
            ),
            (
                f"duty_cycle >= {config.persistent_min_duty_cycle} or "
                f"max_gap_hours <= {config.persistent_max_gap_hours} "
                "(detected on a large share of spanned days, and/or no long silent gap)."
            ),
            (
                f"duty_cycle < {config.persistent_min_duty_cycle} and "
                f"max_gap_hours > {config.persistent_max_gap_hours}, but observed "
                f"over > {config.short_lived_max_duration_hours}h with "
                f">= {config.min_detections_for_classification} detections "
                "(repeated but not continuous/consistent activity)."
            ),
        ],
        default="unclassified",
    )

    return out
