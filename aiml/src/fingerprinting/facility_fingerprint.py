"""
Per-facility historical thermal fingerprint (GIFT Stage I.3).

====================================================================
DESCRIPTIVE BASELINE ONLY -- NOT ANOMALY DETECTION, NOT CLASSIFICATION
====================================================================
This module answers exactly one question, for each normalized OSM
facility:

    "What thermal behaviour has historically been observed
     around/within this facility, according to the confirmed Stage I.2
     spatial associations?"

It does NOT answer:

    "Is a given event unusual for this facility?"          (Stage I.4)
    "Was this thermal event caused by this facility?"      (not in scope)
    "Is this facility an industrial fire / wildfire / etc.?" (not in scope)

No anomaly score, causal claim or source label is computed anywhere in
this module. A facility's fingerprint is simply a robust statistical
summary of its own confirmed historical observations -- nothing is
compared against anything else, and no pseudo-label is generated for any
downstream model to "predict".

WHICH EVENTS COUNT AS A CONFIRMED HISTORICAL OBSERVATION
------------------------------------------------------------------------
Only events where Stage I.2 selected exactly one facility (`facility_id`
is not null -- i.e. `facility_association_method` is `WITHIN_FACILITY`,
`INTERSECTS_FACILITY` or `NEAR_FACILITY`) are used to build a facility's
primary fingerprint statistics. `AMBIGUOUS` events (Stage I.2 explicitly
declined to pick a single facility among multiple similarly-plausible
candidates) and `NO_FACILITY_ASSOCIATION` events are NEVER treated as a
confirmed observation for any facility -- this is deliberately
conservative, per the task's explicit instruction not to silently
attribute an ambiguous event to every candidate. The (informational-only)
`ambiguous_candidate_opportunity_count` column separately records how
often a facility appeared as an *unresolved* candidate for an ambiguous
event, without ever counting it as a confirmed observation.

EVENT-LEVEL VS. DETECTION-LEVEL COUNTS
------------------------------------------------------------------------
`event_count` (number of Stage G thermal events confirmed-associated
with a facility) and `detection_count` (number of underlying FIRMS
detections across those events, from Stage G's own per-event
`detection_count` column) are always kept distinct -- the latter is
summed from the former's constituent events, never counted as
independent "observations" for the `fingerprint_status` rule (a single
166-day persistent event with hundreds of detections is one historical
*episode*, not hundreds of independent sightings).

`detection_count` is derived by SUMMING each confirmed event's own
Stage G `detection_count` field, not by re-reading the 1.17M-row
`thermal_event_detections.csv` -- Stage G's own per-event field already
counts exactly the FIRMS detections that contributed to that event, so
re-reading the detection-level table would recompute an identical number
at a much higher I/O cost. `thermal_event_detections.csv` is
intentionally not read anywhere in Stage I.3.

DAY/NIGHT EVENT CLASSIFICATION
------------------------------------------------------------------------
Stage G already aggregates each event's `day_detection_count`/
`night_detection_count` (from FIRMS `daynight`). This module classifies
each event, deterministically, from those two counts:

    night_detection_count == 0 and day_detection_count > 0   -> "DAY"
    day_detection_count == 0 and night_detection_count > 0   -> "NIGHT"
    day_detection_count > 0 and night_detection_count > 0    -> "MIXED"
    both == 0 (should not occur given Stage G's min_samples>=2,
        but handled defensively)                             -> "UNKNOWN"

`day_event_count`/`night_event_count` only count events classified
"DAY"/"NIGHT" respectively; `MIXED`/`UNKNOWN` events are excluded from
both numerators but remain in the `event_count` denominator used for
`day_event_fraction`/`night_event_fraction`. Consequently these two
fractions are NOT guaranteed to sum to 1 -- a facility whose events are
frequently mixed day/night detections will show fractions that sum to
less than 1, by design (see `README.md` Stage I.3 section).

MONTH-OF-YEAR ATTRIBUTION
------------------------------------------------------------------------
`active_month_count` and the companion monthly profile
(`monthly_profile.py`) attribute each confirmed event to a single
calendar month -- the UTC month of its `event_start` timestamp. A
long-running persistent event that spans several months is attributed
only to its start month, not split/duplicated across every month it
touches. This keeps event-to-month attribution simple, deterministic and
free of double-counting; it is documented here as a known simplification
rather than hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.fingerprinting.fingerprint_config import FingerprintConfig
from src.fingerprinting.robust_stats import grouped_summary_stats

REQUIRED_EVENT_COLUMNS: tuple[str, ...] = (
    "event_id",
    "detection_count",
    "event_start",
    "event_end",
    "observed_duration_hours",
    "distinct_detection_days",
    "peak_frp",
    "day_detection_count",
    "night_detection_count",
    "persistence_label",
    "facility_id",
    "facility_association_method",
    "facility_attribution_confidence",
    "facility_distance_km",
    "candidate_facility_ids",
)

REQUIRED_FACILITY_COLUMNS: tuple[str, ...] = ("facility_id", "facility_name", "facility_type")

DAY = "DAY"
NIGHT = "NIGHT"
MIXED = "MIXED"
UNKNOWN_DAYNIGHT = "UNKNOWN"

# Statistic-family output prefixes and the event column each summarizes.
_SUMMARY_FIELDS: tuple[tuple[str, str], ...] = (
    ("peak_frp", "peak_frp"),
    ("event_size", "detection_count"),
    ("duration_hours", "observed_duration_hours"),
    ("distance_km", "facility_distance_km"),
)

# Every numeric column produced only for confirmed-associated facilities
# that must stay NaN (never fabricated as 0) for a NO_OBSERVATIONS facility.
_STATISTIC_COLUMNS: tuple[str, ...] = tuple(
    f"{prefix}_{suffix}" for prefix, _ in _SUMMARY_FIELDS for suffix in ("median", "mad", "p25", "p75", "p90", "max")
) + ("day_event_fraction", "night_event_fraction", "persistent_event_fraction", "recurring_event_fraction", "short_lived_event_fraction")

# Count-type columns that are legitimately 0 (not missing) for a facility
# with zero confirmed observations.
_COUNT_COLUMNS: tuple[str, ...] = (
    "event_count",
    "detection_count",
    "observation_day_count",
    "active_month_count",
    "day_event_count",
    "night_event_count",
    "persistent_event_count",
    "recurring_event_count",
    "short_lived_event_count",
    "insufficient_observations_event_count",
    "within_facility_count",
    "intersects_facility_count",
    "near_facility_count",
    "high_confidence_event_count",
    "medium_confidence_event_count",
    "low_confidence_event_count",
    "fingerprint_observation_count",
    "ambiguous_candidate_opportunity_count",
)

OUTPUT_COLUMNS: tuple[str, ...] = (
    "facility_id",
    "facility_name",
    "facility_type",
    "event_count",
    "detection_count",
    "observation_day_count",
    "first_observation_date",
    "last_observation_date",
    "observation_span_days",
    "active_month_count",
    "day_event_count",
    "night_event_count",
    "day_event_fraction",
    "night_event_fraction",
    *[f"peak_frp_{suffix}" for suffix in ("median", "mad", "p25", "p75", "p90", "max")],
    *[f"event_size_{suffix}" for suffix in ("median", "mad", "p25", "p75", "p90", "max")],
    *[f"duration_hours_{suffix}" for suffix in ("median", "mad", "p25", "p75", "p90", "max")],
    "persistent_event_count",
    "persistent_event_fraction",
    "recurring_event_count",
    "recurring_event_fraction",
    "short_lived_event_count",
    "short_lived_event_fraction",
    "insufficient_observations_event_count",
    *[f"distance_km_{suffix}" for suffix in ("median", "mad", "p25", "p75", "p90", "max")],
    "within_facility_count",
    "intersects_facility_count",
    "near_facility_count",
    "high_confidence_event_count",
    "medium_confidence_event_count",
    "low_confidence_event_count",
    "ambiguous_candidate_opportunity_count",
    "fingerprint_observation_count",
    "fingerprint_status",
)


def _count_column(counts_df: pd.DataFrame, column: str, index: pd.Index) -> pd.Series:
    """`counts_df[column]` reindexed to `index` (fill 0), or all-zero if absent.

    A thin helper around the `unstack()` result of a groupby -- the
    unstacked DataFrame simply has no column at all for a category that
    never occurs in the data (e.g. no event is ever `INTERSECTS_FACILITY`
    for a small synthetic fixture), which `DataFrame.get` alone cannot
    turn into a zero-filled Series aligned to `index`.
    """
    if column in counts_df.columns:
        return counts_df[column].reindex(index, fill_value=0)
    return pd.Series(0, index=index)


def classify_event_daynight(day_detection_count: pd.Series, night_detection_count: pd.Series) -> pd.Series:
    """Deterministically classify each event as DAY/NIGHT/MIXED/UNKNOWN.

    See module docstring for the exact rule. Fully vectorized.
    """
    day = pd.to_numeric(day_detection_count, errors="coerce").fillna(0)
    night = pd.to_numeric(night_detection_count, errors="coerce").fillna(0)
    return pd.Series(
        np.select(
            [(night == 0) & (day > 0), (day == 0) & (night > 0), (day > 0) & (night > 0)],
            [DAY, NIGHT, MIXED],
            default=UNKNOWN_DAYNIGHT,
        ),
        index=day.index,
    )


def _ambiguous_candidate_opportunity_counts(events_df: pd.DataFrame) -> pd.Series:
    """Count, per facility_id, how many AMBIGUOUS events listed it as a candidate.

    Purely informational (see module docstring) -- never merged into
    `event_count` or any confirmed-observation statistic. Computed from
    `candidate_facility_ids` (already present on the Stage I.2 output),
    so no separate read of `thermal_event_facility_candidates.csv` is
    required.
    """
    ambiguous = events_df.loc[events_df["facility_association_method"] == "AMBIGUOUS", "candidate_facility_ids"]
    if ambiguous.empty:
        return pd.Series(dtype="int64")
    exploded = ambiguous.dropna().str.split(",").explode()
    exploded = exploded[exploded.str.len() > 0]
    if exploded.empty:
        return pd.Series(dtype="int64")
    return exploded.value_counts()


def build_facility_fingerprints(
    events_df: pd.DataFrame, facilities_df: pd.DataFrame, config: FingerprintConfig
) -> pd.DataFrame:
    """Build one fingerprint row per facility in `facilities_df`.

    Args:
        events_df: Stage I.2 output (`thermal_events_with_facility_association.csv`),
            unmodified. Must contain `REQUIRED_EVENT_COLUMNS`.
        facilities_df: Stage I.1 output (`osm_facilities.csv`), unmodified.
            Must contain `REQUIRED_FACILITY_COLUMNS`. Every `facility_id`
            here gets exactly one output row, including facilities with
            zero confirmed associated events.
        config: `FingerprintConfig`.

    Returns:
        A DataFrame with `OUTPUT_COLUMNS`, one row per `facility_id` in
        `facilities_df`, sorted by `facility_id` for determinism.

    Raises:
        ValueError: If required columns are missing.
    """
    missing_event_cols = [c for c in REQUIRED_EVENT_COLUMNS if c not in events_df.columns]
    if missing_event_cols:
        raise ValueError(f"Events table is missing required column(s): {missing_event_cols}.")
    missing_facility_cols = [c for c in REQUIRED_FACILITY_COLUMNS if c not in facilities_df.columns]
    if missing_facility_cols:
        raise ValueError(f"Facilities table is missing required column(s): {missing_facility_cols}.")

    associated = events_df.loc[events_df["facility_id"].notna()].copy()

    ambiguous_counts = _ambiguous_candidate_opportunity_counts(events_df)

    if associated.empty:
        base = facilities_df[list(REQUIRED_FACILITY_COLUMNS)].drop_duplicates("facility_id").copy()
        return _finalize(base, pd.DataFrame(index=pd.Index([], name="facility_id")), ambiguous_counts, config)

    associated["event_start_dt"] = pd.to_datetime(associated["event_start"], utc=True)
    associated["event_end_dt"] = pd.to_datetime(associated["event_end"], utc=True)
    associated["event_month"] = associated["event_start_dt"].dt.month
    associated["daynight_class"] = classify_event_daynight(
        associated["day_detection_count"], associated["night_detection_count"]
    )

    grouped = associated.groupby("facility_id", sort=False)

    stats = pd.DataFrame(index=grouped.size().index)
    stats["event_count"] = grouped.size()
    stats["detection_count"] = grouped["detection_count"].sum()
    stats["observation_day_count"] = grouped["distinct_detection_days"].sum()
    stats["first_observation_date"] = grouped["event_start_dt"].min()
    stats["last_observation_date"] = grouped["event_end_dt"].max()
    stats["observation_span_days"] = (stats["last_observation_date"] - stats["first_observation_date"]).dt.total_seconds() / 86400.0
    stats["active_month_count"] = grouped["event_month"].nunique()

    daynight_counts = associated.groupby(["facility_id", "daynight_class"], sort=False).size().unstack(fill_value=0)
    stats["day_event_count"] = _count_column(daynight_counts, DAY, stats.index)
    stats["night_event_count"] = _count_column(daynight_counts, NIGHT, stats.index)
    stats["day_event_fraction"] = stats["day_event_count"] / stats["event_count"]
    stats["night_event_fraction"] = stats["night_event_count"] / stats["event_count"]

    for prefix, source_col in _SUMMARY_FIELDS:
        stats = stats.join(grouped_summary_stats(associated, "facility_id", source_col, prefix))

    persistence_counts = associated.groupby(["facility_id", "persistence_label"], sort=False).size().unstack(fill_value=0)
    stats["persistent_event_count"] = _count_column(persistence_counts, "PERSISTENT", stats.index)
    stats["recurring_event_count"] = _count_column(persistence_counts, "RECURRING", stats.index)
    stats["short_lived_event_count"] = _count_column(persistence_counts, "SHORT_LIVED", stats.index)
    stats["insufficient_observations_event_count"] = _count_column(
        persistence_counts, "INSUFFICIENT_OBSERVATIONS", stats.index
    )
    stats["persistent_event_fraction"] = stats["persistent_event_count"] / stats["event_count"]
    stats["recurring_event_fraction"] = stats["recurring_event_count"] / stats["event_count"]
    stats["short_lived_event_fraction"] = stats["short_lived_event_count"] / stats["event_count"]

    method_counts = associated.groupby(["facility_id", "facility_association_method"], sort=False).size().unstack(fill_value=0)
    stats["within_facility_count"] = _count_column(method_counts, "WITHIN_FACILITY", stats.index)
    stats["intersects_facility_count"] = _count_column(method_counts, "INTERSECTS_FACILITY", stats.index)
    stats["near_facility_count"] = _count_column(method_counts, "NEAR_FACILITY", stats.index)

    confidence_counts = associated.groupby(["facility_id", "facility_attribution_confidence"], sort=False).size().unstack(fill_value=0)
    stats["high_confidence_event_count"] = _count_column(confidence_counts, "HIGH", stats.index)
    stats["medium_confidence_event_count"] = _count_column(confidence_counts, "MEDIUM", stats.index)
    stats["low_confidence_event_count"] = _count_column(confidence_counts, "LOW", stats.index)

    base = facilities_df[list(REQUIRED_FACILITY_COLUMNS)].drop_duplicates("facility_id").copy()
    return _finalize(base, stats, ambiguous_counts, config)


def _finalize(
    base: pd.DataFrame, stats: pd.DataFrame, ambiguous_counts: pd.Series, config: FingerprintConfig
) -> pd.DataFrame:
    """Left-join per-facility stats onto the full facility universe and fill defaults."""
    merged = base.set_index("facility_id").join(stats, how="left")
    merged = merged.join(ambiguous_counts.rename("ambiguous_candidate_opportunity_count"), how="left")

    for col in _COUNT_COLUMNS:
        if col == "fingerprint_observation_count":
            continue
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype("int64")
        else:
            merged[col] = 0

    for col in _STATISTIC_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan

    merged["fingerprint_observation_count"] = merged["event_count"]
    merged["fingerprint_status"] = merged["event_count"].apply(config.classify_status)

    merged = merged.reset_index()
    for col in OUTPUT_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan
    merged = merged[list(OUTPUT_COLUMNS)]
    return merged.sort_values("facility_id", kind="mergesort").reset_index(drop=True)
