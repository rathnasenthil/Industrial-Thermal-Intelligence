"""
Event-level feature aggregation for GIFT Stage G thermal events.

Given a set of FIRMS detections already assigned to an `event_id` (see
`src.event_formation.st_dbscan`), this module aggregates them into one
row per event: thermal statistics, spatial geometry, temporal span,
FIRMS-confidence counts and day/night counts. Nothing here classifies an
event's source (industrial/wildfire/agricultural/etc.) — that is left to
later GIFT stages.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.event_formation.geometry import compute_event_geometry

# Columns expected on the input (labeled) detections DataFrame.
_REQUIRED_COLUMNS: tuple[str, ...] = (
    "latitude",
    "longitude",
    "acq_datetime",
    "frp",
    "frp_valid",
    "bright_ti4",
    "bright_ti5",
    "confidence",
    "daynight",
)


def _median_absolute_deviation(values: np.ndarray) -> float:
    """Median absolute deviation (MAD) around the median, unscaled."""
    if values.size == 0:
        return float("nan")
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def _confidence_counts(confidence: pd.Series) -> dict[str, int]:
    """Count FIRMS confidence codes without inventing a numeric mapping.

    Handles both the VIIRS letter codes (n/l/h) and leaves any other
    literal value (e.g. a MODIS-style numeric string) out of the
    dedicated n/l/h buckets but still represented in
    `confidence_distribution`.
    """
    counts = confidence.value_counts(dropna=False)
    return {
        "confidence_n_count": int(counts.get("n", 0)),
        "confidence_l_count": int(counts.get("l", 0)),
        "confidence_h_count": int(counts.get("h", 0)),
        "confidence_distribution": {str(k): int(v) for k, v in counts.items()},
    }


def compute_event_row(event_id: str, detections: pd.DataFrame) -> dict:
    """Compute one thermal-event summary row from its member detections.

    Args:
        event_id: The event identifier assigned to this group of
            detections.
        detections: DataFrame of the detections belonging to this event
            (must contain the columns in `_REQUIRED_COLUMNS`).

    Returns:
        A flat dict of event-level fields (see module/requirements for
        the full field list): identity, geometry, thermal statistics,
        temporal span, persistence indicators, confidence counts and
        day/night counts.
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in detections.columns]
    if missing:
        raise ValueError(f"Detections DataFrame is missing required column(s): {missing}")

    lat = detections["latitude"].to_numpy(dtype=np.float64)
    lon = detections["longitude"].to_numpy(dtype=np.float64)
    geom = compute_event_geometry(lat, lon)

    acq_sorted = detections["acq_datetime"].sort_values()
    event_start = acq_sorted.iloc[0]
    event_end = acq_sorted.iloc[-1]
    observed_duration = event_end - event_start
    observed_duration_hours = observed_duration.total_seconds() / 3600.0

    # Persistence indicators: how spread out (in time) are the detections,
    # independent of any claim about the true physical duration.
    distinct_days = int(acq_sorted.dt.date.nunique())
    if len(acq_sorted) > 1:
        gaps_hours = acq_sorted.diff().dropna().dt.total_seconds() / 3600.0
        mean_gap_hours = float(gaps_hours.mean())
        max_gap_hours = float(gaps_hours.max())
    else:
        mean_gap_hours = float("nan")
        max_gap_hours = float("nan")

    detection_count = len(detections)
    detection_frequency_per_day = (
        detection_count / (observed_duration_hours / 24.0) if observed_duration_hours > 0 else float("nan")
    )

    frp_valid = detections.loc[detections["frp_valid"], "frp"].to_numpy(dtype=np.float64)
    if frp_valid.size > 0:
        frp_stats = {
            "peak_frp": float(np.max(frp_valid)),
            "mean_frp": float(np.mean(frp_valid)),
            "median_frp": float(np.median(frp_valid)),
            "frp_std": float(np.std(frp_valid, ddof=0)) if frp_valid.size > 1 else 0.0,
            "frp_mad": _median_absolute_deviation(frp_valid),
            # NOTE: sum of instantaneous per-overpass FRP estimates. This is
            # a rough cumulative signal-strength indicator across observed
            # passes, NOT a physically integrated total radiated energy
            # (that would require integrating power over continuous time,
            # which discrete satellite overpasses cannot provide).
            "total_frp": float(np.sum(frp_valid)),
            "frp_valid_count": int(frp_valid.size),
        }
    else:
        frp_stats = {
            "peak_frp": None,
            "mean_frp": None,
            "median_frp": None,
            "frp_std": None,
            "frp_mad": None,
            "total_frp": None,
            "frp_valid_count": 0,
        }

    bright_ti4 = detections["bright_ti4"].dropna().to_numpy(dtype=np.float64)
    bright_ti5 = detections["bright_ti5"].dropna().to_numpy(dtype=np.float64)

    daynight_counts = detections["daynight"].value_counts(dropna=False)
    confidence_stats = _confidence_counts(detections["confidence"])

    row = {
        "event_id": event_id,
        "detection_count": detection_count,
        # --- geometry ---
        "centroid_latitude": geom.centroid_latitude,
        "centroid_longitude": geom.centroid_longitude,
        "centroid_wkt": geom.centroid_wkt,
        "footprint_wkt": geom.footprint_wkt,
        "min_latitude": geom.min_latitude,
        "max_latitude": geom.max_latitude,
        "min_longitude": geom.min_longitude,
        "max_longitude": geom.max_longitude,
        # --- temporal ---
        "event_start": event_start.isoformat(),
        "event_end": event_end.isoformat(),
        "observed_duration_hours": observed_duration_hours,
        "distinct_detection_days": distinct_days,
        "mean_gap_hours": mean_gap_hours,
        "max_gap_hours": max_gap_hours,
        "detection_frequency_per_day": detection_frequency_per_day,
        # --- thermal ---
        **frp_stats,
        "max_bright_ti4": float(np.max(bright_ti4)) if bright_ti4.size else None,
        "mean_bright_ti4": float(np.mean(bright_ti4)) if bright_ti4.size else None,
        "max_bright_ti5": float(np.max(bright_ti5)) if bright_ti5.size else None,
        "mean_bright_ti5": float(np.mean(bright_ti5)) if bright_ti5.size else None,
        # --- day/night ---
        "day_detection_count": int(daynight_counts.get("D", 0)),
        "night_detection_count": int(daynight_counts.get("N", 0)),
        # --- confidence (native FIRMS codes preserved, no fabricated mapping) ---
        **confidence_stats,
    }
    return row


def build_thermal_events(labeled_detections: pd.DataFrame, event_id_column: str = "event_id") -> pd.DataFrame:
    """Aggregate labeled detections into one row per thermal event.

    This produces exactly the same fields as calling `compute_event_row`
    once per event, but uses vectorized pandas ``groupby`` aggregations
    across *all* events at once (rather than a Python-level loop calling
    several pandas operations per event). That distinction matters at
    scale: a naive per-event loop was measured at ~5-6 ms/event, which
    would take on the order of tens of minutes for the ~10^5 events
    expected from the full ~1.17M-detection dataset. The only part that
    still runs per-event is the convex-hull footprint (`shapely` has no
    batched/vectorized convex-hull-per-group API), which is comparatively
    cheap once it is the *only* thing left in the loop.

    Args:
        labeled_detections: Detections that have already been assigned a
            non-null `event_id` (noise rows must be excluded before
            calling this).
        event_id_column: Name of the event-id column to group by.

    Returns:
        A DataFrame with one row per event (see `compute_event_row` for
        the full column list), sorted by `event_start`.

    Raises:
        ValueError: If required columns are missing.
    """
    if labeled_detections.empty:
        return pd.DataFrame()

    missing = [c for c in _REQUIRED_COLUMNS if c not in labeled_detections.columns]
    if missing:
        raise ValueError(f"Detections DataFrame is missing required column(s): {missing}")

    df = labeled_detections.reset_index(drop=True)
    grouped = df.groupby(event_id_column, sort=False)

    # --- detection count, geometry (all natively vectorized aggregations) ---
    detection_count = grouped.size().rename("detection_count")
    centroid_latitude = grouped["latitude"].mean().rename("centroid_latitude")
    centroid_longitude = grouped["longitude"].mean().rename("centroid_longitude")
    min_latitude = grouped["latitude"].min().rename("min_latitude")
    max_latitude = grouped["latitude"].max().rename("max_latitude")
    min_longitude = grouped["longitude"].min().rename("min_longitude")
    max_longitude = grouped["longitude"].max().rename("max_longitude")

    # --- temporal span (vectorized) ---
    event_start = grouped["acq_datetime"].min().rename("event_start")
    event_end = grouped["acq_datetime"].max().rename("event_end")
    observed_duration_hours = ((event_end - event_start).dt.total_seconds() / 3600.0).rename(
        "observed_duration_hours"
    )
    distinct_detection_days = (
        df.assign(_date=df["acq_datetime"].dt.date).groupby(event_id_column, sort=False)["_date"].nunique()
    ).rename("distinct_detection_days")

    # Per-row gap to the previous detection *within the same event*
    # (vectorized: one sort + one groupby-diff over the whole dataset).
    df_sorted = df.sort_values([event_id_column, "acq_datetime"])
    gap_hours = (
        df_sorted.groupby(event_id_column, sort=False)["acq_datetime"].diff().dt.total_seconds() / 3600.0
    )
    gap_by_event = gap_hours.groupby(df_sorted[event_id_column], sort=False)
    mean_gap_hours = gap_by_event.mean().rename("mean_gap_hours")
    max_gap_hours = gap_by_event.max().rename("max_gap_hours")

    detection_frequency_per_day = (
        detection_count / (observed_duration_hours / 24.0)
    ).where(observed_duration_hours > 0).rename("detection_frequency_per_day")

    # --- FRP statistics (only over frp_valid rows; missing/invalid FRP is
    # never fabricated — events with zero valid FRP get None/NaN, not 0) ---
    valid_frp = df.loc[df["frp_valid"], [event_id_column, "frp"]].copy()
    valid_grouped = valid_frp.groupby(event_id_column, sort=False)["frp"]
    peak_frp = valid_grouped.max().rename("peak_frp")
    mean_frp = valid_grouped.mean().rename("mean_frp")
    median_frp = valid_grouped.median().rename("median_frp")
    frp_std = valid_grouped.std(ddof=0).rename("frp_std")
    total_frp = valid_grouped.sum().rename("total_frp")
    frp_valid_count = valid_grouped.size().rename("frp_valid_count")

    if not valid_frp.empty:
        valid_frp["_group_median"] = valid_frp.groupby(event_id_column, sort=False)["frp"].transform("median")
        valid_frp["_abs_dev"] = (valid_frp["frp"] - valid_frp["_group_median"]).abs()
        frp_mad = valid_frp.groupby(event_id_column, sort=False)["_abs_dev"].median().rename("frp_mad")
    else:
        frp_mad = pd.Series(dtype=float, name="frp_mad")

    # --- brightness temperatures (pandas aggregations skip NaN by default) ---
    max_bright_ti4 = grouped["bright_ti4"].max().rename("max_bright_ti4")
    mean_bright_ti4 = grouped["bright_ti4"].mean().rename("mean_bright_ti4")
    max_bright_ti5 = grouped["bright_ti5"].max().rename("max_bright_ti5")
    mean_bright_ti5 = grouped["bright_ti5"].mean().rename("mean_bright_ti5")

    # --- day/night counts (vectorized pivot) ---
    daynight_pivot = pd.crosstab(df[event_id_column], df["daynight"])
    day_detection_count = daynight_pivot.get("D", pd.Series(0, index=daynight_pivot.index)).rename(
        "day_detection_count"
    )
    night_detection_count = daynight_pivot.get("N", pd.Series(0, index=daynight_pivot.index)).rename(
        "night_detection_count"
    )

    # --- confidence counts (vectorized pivot; distribution dict built once
    # per event, not once per detection) ---
    confidence_pivot = pd.crosstab(df[event_id_column], df["confidence"])
    confidence_n_count = confidence_pivot.get("n", pd.Series(0, index=confidence_pivot.index)).rename(
        "confidence_n_count"
    )
    confidence_l_count = confidence_pivot.get("l", pd.Series(0, index=confidence_pivot.index)).rename(
        "confidence_l_count"
    )
    confidence_h_count = confidence_pivot.get("h", pd.Series(0, index=confidence_pivot.index)).rename(
        "confidence_h_count"
    )
    confidence_distribution = confidence_pivot.apply(
        lambda row: {str(k): int(v) for k, v in row.items() if v > 0}, axis=1
    ).rename("confidence_distribution")

    events_df = pd.concat(
        [
            detection_count,
            centroid_latitude,
            centroid_longitude,
            min_latitude,
            max_latitude,
            min_longitude,
            max_longitude,
            event_start,
            event_end,
            observed_duration_hours,
            distinct_detection_days,
            mean_gap_hours,
            max_gap_hours,
            detection_frequency_per_day,
            peak_frp,
            mean_frp,
            median_frp,
            frp_std,
            frp_mad,
            total_frp,
            frp_valid_count,
            max_bright_ti4,
            mean_bright_ti4,
            max_bright_ti5,
            mean_bright_ti5,
            day_detection_count,
            night_detection_count,
            confidence_n_count,
            confidence_l_count,
            confidence_h_count,
            confidence_distribution,
        ],
        axis=1,
    )

    events_df["frp_valid_count"] = events_df["frp_valid_count"].fillna(0).astype(int)
    events_df["detection_count"] = events_df["detection_count"].astype(int)
    for count_col in ("day_detection_count", "night_detection_count", "confidence_n_count", "confidence_l_count", "confidence_h_count"):
        events_df[count_col] = events_df[count_col].fillna(0).astype(int)
    events_df["confidence_distribution"] = events_df["confidence_distribution"].apply(
        lambda v: v if isinstance(v, dict) else {}
    )

    # Events with zero valid FRP: leave stats as None (never fabricate 0).
    no_valid_frp = events_df["frp_valid_count"] == 0
    for col in ("peak_frp", "mean_frp", "median_frp", "frp_std", "frp_mad", "total_frp"):
        events_df.loc[no_valid_frp, col] = None
    # A single valid FRP observation has an undefined (not zero) std/MAD
    # in the per-event compute_event_row reference implementation only
    # when there's exactly one sample; pandas' ddof=0 std of one value is
    # already 0.0 and MAD of one value is already 0.0, which matches.

    # --- footprint geometry: the one genuinely per-event step (shapely
    # has no vectorized/batched convex-hull-per-group operation) ---
    footprint_wkt = {}
    centroid_wkt = {}
    for event_id, group in df.groupby(event_id_column, sort=False):
        geom = compute_event_geometry(
            group["latitude"].to_numpy(dtype=np.float64), group["longitude"].to_numpy(dtype=np.float64)
        )
        footprint_wkt[event_id] = geom.footprint_wkt
        centroid_wkt[event_id] = geom.centroid_wkt

    events_df["footprint_wkt"] = events_df.index.map(footprint_wkt)
    events_df["centroid_wkt"] = events_df.index.map(centroid_wkt)

    events_df["event_id"] = events_df.index
    events_df["event_start"] = events_df["event_start"].apply(lambda ts: ts.isoformat())
    events_df["event_end"] = events_df["event_end"].apply(lambda ts: ts.isoformat())

    column_order = [
        "event_id",
        "detection_count",
        "centroid_latitude",
        "centroid_longitude",
        "centroid_wkt",
        "footprint_wkt",
        "min_latitude",
        "max_latitude",
        "min_longitude",
        "max_longitude",
        "event_start",
        "event_end",
        "observed_duration_hours",
        "distinct_detection_days",
        "mean_gap_hours",
        "max_gap_hours",
        "detection_frequency_per_day",
        "peak_frp",
        "mean_frp",
        "median_frp",
        "frp_std",
        "frp_mad",
        "total_frp",
        "frp_valid_count",
        "max_bright_ti4",
        "mean_bright_ti4",
        "max_bright_ti5",
        "mean_bright_ti5",
        "day_detection_count",
        "night_detection_count",
        "confidence_n_count",
        "confidence_l_count",
        "confidence_h_count",
        "confidence_distribution",
    ]
    events_df = events_df[column_order].reset_index(drop=True)
    return events_df.sort_values("event_start").reset_index(drop=True)
