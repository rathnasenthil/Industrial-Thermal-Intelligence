"""
Candidate ranking and association-decision logic for GIFT Stage I.2.

====================================================================
FACILITY ASSOCIATION IS NOT SOURCE CLASSIFICATION -- READ THIS FIRST
====================================================================
This module answers exactly one question:

    "Which normalized OSM facility record(s), if any, are spatially
     plausible neighbors of this thermal event, and how confident is
     that *spatial* match?"

It does NOT, and must never be extended by later stages reading its
output to, answer:

    "What caused this thermal event?"

A `WITHIN_FACILITY` result means the event's centroid falls inside a
facility's mapped OSM polygon -- nothing more. It is not proof the
facility caused the thermal event; OSM facility polygons vary hugely in
completeness/precision, a thermal event can occur near a facility for
reasons unrelated to it (a field fire adjacent to a refinery boundary is
still a field fire), and OSM itself is contextual, crowd-sourced
evidence, not a ground-truth industrial-activity registry (see
`src.infrastructure.facility_report` for the same caveat applied to
Stage I.1). No field produced anywhere in Stage I.2 is named or
documented as a source/cause classification, and none should be
interpreted as one. That determination -- if ever made -- belongs to a
later GIFT stage this module does not implement.

RANKING / SELECTION ALGORITHM
------------------------------------------------------------------------
Given the raw candidate pairs from `association_geometry.find_candidate_pairs`,
candidates for each event are ranked by, in strict priority order:

    1. spatial relationship tier: WITHIN_FACILITY > INTERSECTS_FACILITY > NEAR_FACILITY
    2. distance_km (ascending -- closer is a better match)
    3. facility geometry quality (Polygon/MultiPolygon ranked above Point --
       a polygon's own boundary is a more precise spatial signal than a
       single representative point, all else equal)
    4. facility_type (lexical order -- an arbitrary but deterministic and
       explicitly NON-judgmental tie-break; this is NOT a claim that any
       facility_type is "more likely" to be a source)
    5. facility_id (lexical order -- final deterministic tie-break; every
       facility_id is unique, so this always fully resolves ties)

The top-ranked candidate is the event's tentative association UNLESS it
cannot be confidently distinguished from the runner-up (same spatial-
relation tier AND within `ambiguity_distance_tolerance_km` of it), in
which case the whole event is marked AMBIGUOUS and NO single facility is
selected in the main output -- see `select_association` for exactly what
is/isn't populated in that case. This deliberately implements the
project's explicit "do not blindly select the nearest facility" rule:
proximity ranking is only ever used to pick among genuinely distinguishable
candidates, never to force a choice between two similarly-plausible ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.infrastructure.association_config import AssociationConfig
from src.infrastructure.association_geometry import INTERSECTS_FACILITY, NEAR_FACILITY, WITHIN_FACILITY

AMBIGUOUS = "AMBIGUOUS"
NO_FACILITY_ASSOCIATION = "NO_FACILITY_ASSOCIATION"

METHODS: tuple[str, ...] = (
    WITHIN_FACILITY,
    INTERSECTS_FACILITY,
    NEAR_FACILITY,
    AMBIGUOUS,
    NO_FACILITY_ASSOCIATION,
)

CONFIDENCE_HIGH = "HIGH"
CONFIDENCE_MEDIUM = "MEDIUM"
CONFIDENCE_LOW = "LOW"
CONFIDENCE_NONE = "NONE"

CONFIDENCE_LEVELS: tuple[str, ...] = (CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW, CONFIDENCE_NONE)

# Lower = higher priority (better spatial evidence). Used for deterministic
# ranking only -- not a claim about likelihood of causation.
_RELATION_PRIORITY: dict[str, int] = {WITHIN_FACILITY: 0, INTERSECTS_FACILITY: 1, NEAR_FACILITY: 2}

# Lower = preferred (a polygon boundary is a more precise spatial signal
# than a single representative point), used only as a deterministic
# tie-break -- see module docstring.
_GEOMETRY_QUALITY_RANK: dict[str, int] = {"Polygon": 0, "MultiPolygon": 0, "Point": 1}

MAIN_OUTPUT_COLUMNS: tuple[str, ...] = (
    "facility_id",
    "facility_name",
    "facility_type",
    "facility_association_method",
    "facility_attribution_confidence",
    "facility_distance_km",
    "candidate_facility_count",
    "candidate_facility_ids",
)

CANDIDATES_OUTPUT_COLUMNS: tuple[str, ...] = (
    "event_id",
    "facility_id",
    "facility_name",
    "facility_type",
    "spatial_relation",
    "distance_km",
    "candidate_rank",
    "candidate_score",
)


def _candidate_score(relation_priority: pd.Series, distance_km: pd.Series) -> pd.Series:
    """Deterministic ranking score -- higher is a better candidate.

    NOT a probability and not statistically calibrated: it exists solely
    to produce a single sortable number consistent with the documented
    ranking priority (relation tier first, distance second). The tier
    gap (10,000) is chosen simply to be far larger than any realistic
    `distance_km` value in this dataset (India's own extent is
    ~3,000 km), guaranteeing tier always dominates the score.
    """
    return (2 - relation_priority) * 10_000.0 - distance_km


def rank_candidates(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Deterministically rank every event's candidate facilities.

    Args:
        pairs_df: Output of `association_geometry.find_candidate_pairs`
            (one row per raw candidate pair).

    Returns:
        A copy of `pairs_df` with `candidate_rank` (1 = best, per event)
        and `candidate_score` columns added, sorted by
        (``event_id``, ``candidate_rank``). Empty input -> empty output
        with the same added columns.
    """
    if pairs_df.empty:
        out = pairs_df.copy()
        out["candidate_rank"] = pd.Series(dtype="int64")
        out["candidate_score"] = pd.Series(dtype="float64")
        return out

    df = pairs_df.copy()
    df["_relation_priority"] = df["spatial_relation"].map(_RELATION_PRIORITY)
    df["_geometry_quality_rank"] = df["geometry_type"].map(_GEOMETRY_QUALITY_RANK).fillna(1).astype(int)
    df["candidate_score"] = _candidate_score(df["_relation_priority"], df["distance_km"])

    # Deterministic sort: relation tier -> distance -> geometry quality ->
    # facility_type (lexical, non-judgmental) -> facility_id (unique,
    # fully resolves any remaining tie). `kind="mergesort"` is stable,
    # but since facility_id is unique this sort has no remaining ties
    # regardless of algorithm.
    df = df.sort_values(
        by=["event_id", "_relation_priority", "distance_km", "_geometry_quality_rank", "facility_type", "facility_id"],
        ascending=True,
        kind="mergesort",
    )
    df["candidate_rank"] = df.groupby("event_id", sort=False).cumcount() + 1
    df = df.drop(columns=["_relation_priority", "_geometry_quality_rank"])
    return df.reset_index(drop=True)


def select_association(all_event_ids: pd.Series, ranked_pairs_df: pd.DataFrame, config: AssociationConfig) -> pd.DataFrame:
    """Select the (if any) confidently-associated facility for every event.

    Fully vectorized (no per-event Python loop): this must scale to the
    production dataset (~179,740 events), where a naive per-event loop
    over pandas groups would be needlessly slow.

    Args:
        all_event_ids: Every `event_id` in the input event table (used to
            guarantee every event gets a row -- including events with
            zero candidates -- so no thermal event is ever dropped by
            this stage).
        ranked_pairs_df: Output of `rank_candidates`.
        config: `AssociationConfig` (uses `ambiguity_distance_tolerance_km`).

    Returns:
        A DataFrame in the same order as `all_event_ids`, with columns
        `MAIN_OUTPUT_COLUMNS` plus ``event_id``.
    """
    base = pd.DataFrame({"event_id": pd.Series(all_event_ids).to_numpy()})

    if ranked_pairs_df.empty:
        base["facility_id"] = None
        base["facility_name"] = None
        base["facility_type"] = None
        base["facility_association_method"] = NO_FACILITY_ASSOCIATION
        base["facility_attribution_confidence"] = CONFIDENCE_NONE
        base["facility_distance_km"] = np.nan
        base["candidate_facility_count"] = 0
        base["candidate_facility_ids"] = ""
        return base[["event_id", *MAIN_OUTPUT_COLUMNS]]

    # Per-event aggregate stats (candidate count, sorted id list) --
    # vectorized groupby, not a Python loop.
    grouped = ranked_pairs_df.groupby("event_id", sort=False)
    candidate_count = grouped.size().rename("candidate_facility_count")
    candidate_ids = grouped["facility_id"].apply(lambda s: ",".join(sorted(s))).rename("candidate_facility_ids")

    rank1 = ranked_pairs_df.loc[ranked_pairs_df["candidate_rank"] == 1].set_index("event_id")
    rank2 = ranked_pairs_df.loc[ranked_pairs_df["candidate_rank"] == 2].set_index("event_id")

    merged = base.set_index("event_id").join(candidate_count).join(candidate_ids)
    merged["candidate_facility_count"] = merged["candidate_facility_count"].fillna(0).astype(int)
    merged["candidate_facility_ids"] = merged["candidate_facility_ids"].fillna("")

    merged = merged.join(rank1[["facility_id", "facility_name", "facility_type", "spatial_relation", "distance_km"]])
    merged = merged.join(rank2[["spatial_relation", "distance_km"]], rsuffix="_2")

    has_second = merged["spatial_relation_2"].notna()
    same_tier = has_second & (merged["spatial_relation"] == merged["spatial_relation_2"])
    close = has_second & ((merged["distance_km"] - merged["distance_km_2"]).abs() <= config.ambiguity_distance_tolerance_km)
    ambiguous = same_tier & close

    has_candidate = merged["candidate_facility_count"] > 0
    is_near = merged["spatial_relation"] == NEAR_FACILITY
    is_within_or_intersects = merged["spatial_relation"].isin([WITHIN_FACILITY, INTERSECTS_FACILITY])

    method = np.select(
        [~has_candidate, ambiguous, is_within_or_intersects, is_near],
        [NO_FACILITY_ASSOCIATION, AMBIGUOUS, merged["spatial_relation"], NEAR_FACILITY],
        default=NO_FACILITY_ASSOCIATION,
    )
    confidence = np.select(
        [~has_candidate, ambiguous, is_within_or_intersects, is_near & (merged["candidate_facility_count"] == 1)],
        [CONFIDENCE_NONE, CONFIDENCE_LOW, CONFIDENCE_HIGH, CONFIDENCE_MEDIUM],
        default=CONFIDENCE_LOW,  # NEAR_FACILITY with >1 candidate
    )

    # No single facility is confidently attributed for AMBIGUOUS or
    # no-candidate events -- see module docstring ("do not blindly select
    # the nearest facility"). Candidate detail is never lost; it remains
    # in thermal_event_facility_candidates.csv regardless.
    suppress = ambiguous | (~has_candidate)
    facility_id = merged["facility_id"].where(~suppress, None)
    facility_name = merged["facility_name"].where(~suppress, None)
    facility_type = merged["facility_type"].where(~suppress, None)
    distance_km = merged["distance_km"].where(~suppress, np.nan).round(6)

    result = pd.DataFrame(
        {
            "event_id": merged.index.to_numpy(),
            "facility_id": facility_id.to_numpy(),
            "facility_name": facility_name.to_numpy(),
            "facility_type": facility_type.to_numpy(),
            "facility_association_method": method,
            "facility_attribution_confidence": confidence,
            "facility_distance_km": distance_km.to_numpy(),
            "candidate_facility_count": merged["candidate_facility_count"].to_numpy(),
            "candidate_facility_ids": merged["candidate_facility_ids"].to_numpy(),
        }
    )
    return result
