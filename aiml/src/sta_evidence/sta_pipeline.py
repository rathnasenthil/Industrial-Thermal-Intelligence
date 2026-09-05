"""
End-to-end NASA STA evidence integration pipeline (GIFT Stage I.5).

Appends STA evidence to Stage I.4 events without modifying I.4 anomaly fields.
Does not perform source classification, risk scoring, or ML.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.sta_evidence.config import STAConfig
from src.sta_evidence.sta_loader import STASourceMissingError, load_all_sta_layers
from src.sta_evidence.sta_matching import build_event_geometries, find_sta_candidate_pairs
from src.sta_evidence.sta_normalization import canonical_to_geodataframe, normalize_sta_geodataframe
from src.sta_evidence.sta_ranking import rank_sta_candidates, select_primary_sta_association
from src.sta_evidence.sta_report import build_sta_report

# I.4 fields that must remain byte-identical in meaning (never recalculated).
I4_IMMUTABLE_COLUMNS: tuple[str, ...] = (
    "anomaly_score",
    "anomaly_status",
    "anomaly_confidence",
    "peak_frp_deviation",
    "event_size_deviation",
    "duration_deviation",
    "distance_deviation",
    "persistence_deviation",
    "monthly_deviation",
)

I5_APPEND_COLUMNS: tuple[str, ...] = (
    "sta_association_status",
    "primary_sta_id",
    "sta_layer_type",
    "sta_match_count",
    "sta_nearest_distance_km",
    "sta_intersection_area_m2",
    "sta_evidence_available",
    "sta_temporal_relation",
    "sta_evidence_quality",
)

FORBIDDEN_OUTPUT_SUBSTRINGS: tuple[str, ...] = (
    "industrial_fire",
    "wildfire_probability",
    "agricultural",
    "source_class",
    "fire_type",
    "industrial_probability",
    "industrial_confidence",
    "risk_score",
)


@dataclass
class STAResult:
    events_df: pd.DataFrame
    candidates_df: pd.DataFrame
    sta_normalized_df: pd.DataFrame
    report: dict[str, Any]


def load_events(path: str | Path) -> pd.DataFrame:
    events_path = Path(path)
    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")
    df = pd.read_csv(events_path)
    required = ("event_id", "event_start", "event_end", "footprint_wkt", "centroid_latitude", "centroid_longitude")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"'{events_path}' missing required column(s): {missing}. "
            "Stage I.5 expects thermal_events_with_anomaly_detection.csv (I.4 output)."
        )
    return df


def run_sta_integration(
    events_df: pd.DataFrame,
    config: STAConfig,
    *,
    events_input_path: str = "<in-memory>",
    sta_gdf=None,
    load_stats: dict[str, Any] | None = None,
) -> STAResult:
    """Run Stage I.5.

    Args:
        events_df: Stage I.4 events (unmodified copy is used).
        config: STAConfig.
        sta_gdf: Optional pre-loaded STA GeoDataFrame (for tests). When None,
            loads from ``config.mask_path`` / ``detection_path``.
        load_stats: Optional load statistics when ``sta_gdf`` is provided.
    """
    start = time.perf_counter()
    working = events_df.copy()

    if sta_gdf is None:
        raw_gdf, load_stats = load_all_sta_layers(config)
    else:
        load_stats = load_stats or {"files_loaded": [], "total_records_read": int(len(sta_gdf)), "layer_types_present": []}
        raw_gdf = sta_gdf

    sta_normalized_df, validation_stats = normalize_sta_geodataframe(raw_gdf, config)
    sta_valid_gdf = canonical_to_geodataframe(sta_normalized_df)

    events_gdf = build_event_geometries(working)
    candidates = find_sta_candidate_pairs(events_gdf, sta_valid_gdf, config)
    ranked = rank_sta_candidates(candidates, config)
    association = select_primary_sta_association(working["event_id"], ranked, config)

    association = association.set_index("event_id").reindex(working["event_id"].astype(str)).reset_index()
    for col in I5_APPEND_COLUMNS:
        working[col] = association[col].to_numpy()

    # Restore dtypes / fill defaults for unmatched
    working["sta_match_count"] = pd.to_numeric(working["sta_match_count"], errors="coerce").fillna(0).astype("int64")
    working["sta_evidence_available"] = working["sta_evidence_available"].fillna(False).astype(bool)

    working = working.sort_values("event_id", kind="mergesort").reset_index(drop=True)
    ranked = ranked.sort_values(["event_id", "candidate_rank"], kind="mergesort").reset_index(drop=True)

    # Invariants
    assert len(working) == len(events_df)
    assert working["event_id"].is_unique

    original_sorted = events_df.copy()
    original_sorted["event_id"] = original_sorted["event_id"].astype(str)
    original_sorted = original_sorted.sort_values("event_id", kind="mergesort").reset_index(drop=True)
    for col in I4_IMMUTABLE_COLUMNS:
        if col not in original_sorted.columns:
            continue
        left = working[col]
        right = original_sorted[col]
        if pd.api.types.is_numeric_dtype(right):
            assert pd.Series(left).reset_index(drop=True).equals(
                pd.Series(right).reset_index(drop=True)
            ) or (
                pd.to_numeric(left, errors="coerce").fillna(-1e18).reset_index(drop=True)
                == pd.to_numeric(right, errors="coerce").fillna(-1e18).reset_index(drop=True)
            ).all()
        else:
            assert left.astype(str).fillna("").reset_index(drop=True).equals(
                right.astype(str).fillna("").reset_index(drop=True)
            )

    cols_lower = " ".join(working.columns).lower()
    for term in FORBIDDEN_OUTPUT_SUBSTRINGS:
        assert term not in cols_lower, f"Forbidden classification field leaked: {term}"

    processing_seconds = time.perf_counter() - start
    report = build_sta_report(
        config=config,
        events_input_path=events_input_path,
        output_df=working,
        candidates_df=ranked,
        load_stats=load_stats,
        validation_stats=validation_stats,
        processing_seconds=processing_seconds,
    )

    # Candidate output: keep ranked columns only
    candidate_cols = [
        "event_id",
        "sta_id",
        "sta_layer_type",
        "relationship",
        "distance_km",
        "intersection_area_m2",
        "sta_temporal_relation",
        "candidate_rank",
    ]
    for c in candidate_cols:
        if c not in ranked.columns:
            ranked[c] = None
    candidates_out = ranked[candidate_cols]

    return STAResult(
        events_df=working,
        candidates_df=candidates_out,
        sta_normalized_df=sta_normalized_df,
        report=report,
    )


def save_outputs(
    result: STAResult,
    *,
    events_output_path: str | Path,
    candidates_output_path: str | Path,
    sta_normalized_output_path: str | Path,
) -> None:
    Path(events_output_path).parent.mkdir(parents=True, exist_ok=True)
    result.events_df.to_csv(events_output_path, index=False)
    result.candidates_df.to_csv(candidates_output_path, index=False)
    # Normalized STA: CSV for compatibility; geometry as WKT.
    result.sta_normalized_df.to_csv(sta_normalized_output_path, index=False)
