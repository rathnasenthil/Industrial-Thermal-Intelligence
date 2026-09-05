"""
Deterministic STA candidate ranking and primary association selection.

Ranking order (documented):
  1. relationship tier: STA_INTERSECTS_EVENT before STA_NEAR_EVENT
  2. smaller distance_km
  3. larger intersection_area_m2
  4. layer priority (MASK before DETECTION by default)
  5. sta_id (lexical)

If the top two candidates share the same relationship tier and their
distances differ by <= ambiguity_distance_tolerance_km, status is
AMBIGUOUS and no primary_sta_id is selected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.sta_evidence.config import (
    NO_STA_ASSOCIATION,
    QUALITY_HIGH,
    QUALITY_LOW,
    QUALITY_MEDIUM,
    QUALITY_NONE,
    STA_AMBIGUOUS,
    STA_ASSOCIATED,
    STA_INTERSECTS_EVENT,
    STA_NEAR_EVENT,
    STAConfig,
)

RELATION_RANK = {STA_INTERSECTS_EVENT: 0, STA_NEAR_EVENT: 1}


def rank_sta_candidates(candidates_df: pd.DataFrame, config: STAConfig) -> pd.DataFrame:
    """Add candidate_rank (1 = best) per event_id. Deterministic mergesort."""
    if candidates_df.empty:
        out = candidates_df.copy()
        out["candidate_rank"] = pd.Series(dtype="int64")
        out["layer_priority"] = pd.Series(dtype="int64")
        return out

    out = candidates_df.copy()
    out["relation_rank"] = out["relationship"].map(RELATION_RANK).fillna(99).astype(int)
    out["layer_priority"] = out["sta_layer_type"].map(lambda x: config.layer_priority.get(str(x), 99)).astype(int)
    out["distance_km"] = pd.to_numeric(out["distance_km"], errors="coerce")
    out["intersection_area_m2"] = pd.to_numeric(out["intersection_area_m2"], errors="coerce")
    # Sort key: smaller distance better; larger intersection better → sort by -area
    out["_neg_area"] = (-out["intersection_area_m2"].fillna(-1.0)).astype(float)
    out = out.sort_values(
        ["event_id", "relation_rank", "distance_km", "_neg_area", "layer_priority", "sta_id"],
        kind="mergesort",
    )
    out["candidate_rank"] = out.groupby("event_id", sort=False).cumcount() + 1
    if config.max_candidates_per_event is not None:
        out = out.loc[out["candidate_rank"] <= config.max_candidates_per_event].copy()
    return out.drop(columns=["_neg_area"]).reset_index(drop=True)


def select_primary_sta_association(
    all_event_ids: pd.Series,
    ranked_candidates: pd.DataFrame,
    config: STAConfig,
) -> pd.DataFrame:
    """Build one association summary row per event_id."""
    event_ids = all_event_ids.astype(str)
    base = pd.DataFrame({"event_id": event_ids.to_numpy()})
    base["sta_association_status"] = NO_STA_ASSOCIATION
    base["primary_sta_id"] = None
    base["sta_layer_type"] = None
    base["sta_match_count"] = 0
    base["sta_nearest_distance_km"] = np.nan
    base["sta_intersection_area_m2"] = np.nan
    base["sta_temporal_relation"] = None
    base["sta_evidence_available"] = False
    base["sta_evidence_quality"] = QUALITY_NONE

    if ranked_candidates.empty:
        return base

    counts = ranked_candidates.groupby("event_id").size().rename("sta_match_count")
    tops = ranked_candidates.loc[ranked_candidates["candidate_rank"] == 1].set_index("event_id")
    seconds = ranked_candidates.loc[ranked_candidates["candidate_rank"] == 2].set_index("event_id")

    summaries: list[dict] = []
    for event_id in tops.index:
        top = tops.loc[event_id]
        if isinstance(top, pd.DataFrame):
            top = top.iloc[0]
        status = STA_ASSOCIATED
        primary = top["sta_id"]
        if event_id in seconds.index:
            second = seconds.loc[event_id]
            if isinstance(second, pd.DataFrame):
                second = second.iloc[0]
            same_tier = top["relationship"] == second["relationship"]
            d1 = float(top["distance_km"]) if pd.notna(top["distance_km"]) else 0.0
            d2 = float(second["distance_km"]) if pd.notna(second["distance_km"]) else 0.0
            if same_tier and abs(d1 - d2) <= config.ambiguity_distance_tolerance_km:
                status = STA_AMBIGUOUS
                primary = None

        quality = classify_evidence_quality(
            status=status,
            relationship=str(top["relationship"]),
            distance_km=float(top["distance_km"]) if pd.notna(top["distance_km"]) else None,
            config=config,
        )
        summaries.append(
            {
                "event_id": event_id,
                "sta_association_status": status,
                "primary_sta_id": primary,
                "sta_layer_type": top["sta_layer_type"] if status != STA_AMBIGUOUS else None,
                "sta_match_count": int(counts.loc[event_id]),
                "sta_nearest_distance_km": float(top["distance_km"]) if pd.notna(top["distance_km"]) else np.nan,
                "sta_intersection_area_m2": (
                    float(top["intersection_area_m2"]) if pd.notna(top.get("intersection_area_m2")) else np.nan
                ),
                "sta_temporal_relation": top.get("sta_temporal_relation"),
                "sta_evidence_available": True,
                "sta_evidence_quality": quality,
            }
        )

    if not summaries:
        return base

    summary_df = pd.DataFrame(summaries).set_index("event_id")
    base = base.set_index("event_id")
    base.update(summary_df)
    # update skips NaNs; force overwrite for key fields from summary
    for col in summary_df.columns:
        base.loc[summary_df.index, col] = summary_df[col]
    return base.reset_index()


def classify_evidence_quality(
    *,
    status: str,
    relationship: str,
    distance_km: float | None,
    config: STAConfig,
) -> str:
    """Engineering evidence-quality categories — not fire probability."""
    if status == NO_STA_ASSOCIATION:
        return QUALITY_NONE
    if status == STA_AMBIGUOUS:
        return QUALITY_LOW
    if relationship == STA_INTERSECTS_EVENT:
        return QUALITY_HIGH
    if relationship == STA_NEAR_EVENT:
        if distance_km is None:
            return QUALITY_LOW
        # Closer half of the radius → MEDIUM; outer half → LOW
        if distance_km <= config.association_radius_km * 0.5:
            return QUALITY_MEDIUM
        return QUALITY_LOW
    return QUALITY_LOW
