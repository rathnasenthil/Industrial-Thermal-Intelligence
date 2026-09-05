"""
Facility monthly thermal activity profile (GIFT Stage I.3).

A compact, normalized (long-format) companion table to
`facility_thermal_fingerprints.csv`, kept separate rather than widening
the main fingerprint table with up to 12 extra columns per facility
(most facilities have confirmed activity in only a handful of months).

SCOPE: this table only describes *historical* month-of-year activity. It
deliberately does NOT compare a facility's current/latest activity
against this profile to flag anything as unusual -- that comparison is
Stage I.4 (not implemented here). Rows only exist for
(facility_id, month) combinations that actually had at least one
confirmed associated event; facilities with zero confirmed events
contribute no rows at all (their `NO_OBSERVATIONS` status is fully
captured in the main fingerprint table).

Each confirmed event is attributed to exactly one calendar month -- the
UTC month of its `event_start` -- never split/duplicated across every
month a long-running event happens to span. See
`facility_fingerprint.py` module docstring for the same design choice
applied consistently across Stage I.3.
"""

from __future__ import annotations

import pandas as pd

OUTPUT_COLUMNS: tuple[str, ...] = ("facility_id", "month", "event_count", "detection_count", "event_fraction")


def build_monthly_profile(events_df: pd.DataFrame) -> pd.DataFrame:
    """Build the (facility_id, month) long-format monthly activity table.

    Args:
        events_df: Stage I.2 output. Must contain `facility_id`,
            `event_start`, `detection_count`.

    Returns:
        A DataFrame with `OUTPUT_COLUMNS`, sorted by
        (`facility_id`, `month`) for determinism. Empty (but correctly
        shaped) if no event has a confirmed facility association.
    """
    required = ("facility_id", "event_start", "detection_count")
    missing = [c for c in required if c not in events_df.columns]
    if missing:
        raise ValueError(f"Events table is missing required column(s): {missing}.")

    associated = events_df.loc[events_df["facility_id"].notna()].copy()
    if associated.empty:
        return pd.DataFrame(columns=list(OUTPUT_COLUMNS))

    associated["month"] = pd.to_datetime(associated["event_start"], utc=True).dt.month

    grouped = associated.groupby(["facility_id", "month"], sort=False)
    profile = grouped.agg(event_count=("event_start", "size"), detection_count=("detection_count", "sum")).reset_index()

    total_events_per_facility = profile.groupby("facility_id")["event_count"].transform("sum")
    profile["event_fraction"] = profile["event_count"] / total_events_per_facility

    profile = profile.sort_values(["facility_id", "month"], kind="mergesort").reset_index(drop=True)
    return profile[list(OUTPUT_COLUMNS)]
