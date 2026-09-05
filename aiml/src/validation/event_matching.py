"""Independent validation event matching (separate from I.2 facility association)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from src.infrastructure.association_geometry import INDIA_EQUAL_AREA_CRS
from src.validation.config import (
    INVALID_REFERENCE,
    MATCHED,
    MULTIPLE_POSSIBLE_MATCHES,
    NO_EVENT_MATCH,
    ValidationConfig,
)
from src.validation.validation_schema import MATCH_OUTPUT_COLUMNS, clean_text, empty_matches_frame

EARTH_RADIUS_KM = 6371.0088


def _parse_event_times(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["event_id"] = out["event_id"].astype(str)
    if "event_start" in out.columns:
        out["_start"] = pd.to_datetime(out["event_start"], utc=True, errors="coerce")
    else:
        out["_start"] = pd.NaT
    if "event_end" in out.columns:
        out["_end"] = pd.to_datetime(out["event_end"], utc=True, errors="coerce")
    else:
        out["_end"] = out["_start"]
    return out


def _haversine_ball_tree(lat: np.ndarray, lon: np.ndarray) -> BallTree:
    coords = np.radians(np.column_stack([lat, lon]))
    return BallTree(coords, metric="haversine")


def match_references_to_events(
    references: pd.DataFrame,
    events: pd.DataFrame,
    config: ValidationConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Spatially/temporally match independent references to thermal events.

    Uses a BallTree haversine index — never builds a dense N×M matrix.
    """
    stats: dict[str, Any] = {
        "matched": 0,
        "multiple_possible_matches": 0,
        "no_event_match": 0,
        "invalid_reference": 0,
        "spatial_tolerance_km": config.spatial_tolerance_km,
        "temporal_tolerance_hours": config.temporal_tolerance_hours,
        "computation_crs_note": INDIA_EQUAL_AREA_CRS,
        "matching_method": "haversine_balltree_plus_temporal_window",
    }
    if references.empty or events.empty:
        return empty_matches_frame(), stats

    ev = _parse_event_times(events)
    required_lat = "centroid_latitude" if "centroid_latitude" in ev.columns else None
    required_lon = "centroid_longitude" if "centroid_longitude" in ev.columns else None
    if required_lat is None or required_lon is None:
        raise ValueError("Events table must include centroid_latitude/centroid_longitude.")

    lat = pd.to_numeric(ev[required_lat], errors="coerce").to_numpy()
    lon = pd.to_numeric(ev[required_lon], errors="coerce").to_numpy()
    valid_ev = np.isfinite(lat) & np.isfinite(lon)
    ev = ev.loc[valid_ev].reset_index(drop=True)
    lat = lat[valid_ev]
    lon = lon[valid_ev]
    tree = _haversine_ball_tree(lat, lon)
    radius_rad = float(config.spatial_tolerance_km) / EARTH_RADIUS_KM

    candidate_cols = [
        c
        for c in (
            "source_intelligence_candidate",
            "evidence_strength",
            "industrial_evidence_score",
        )
        if c in ev.columns
    ]

    rows: list[dict[str, Any]] = []
    for _, ref in references.iterrows():
        vid = clean_text(ref.get("validation_id"), "unknown")
        rlat = pd.to_numeric(ref.get("reference_latitude"), errors="coerce")
        rlon = pd.to_numeric(ref.get("reference_longitude"), errors="coerce")
        if not np.isfinite(rlat) or not np.isfinite(rlon):
            stats["invalid_reference"] += 1
            rows.append(
                {
                    "validation_id": vid,
                    "event_id": None,
                    "reference_label_raw": clean_text(ref.get("reference_label_raw")),
                    "reference_label_normalized": clean_text(ref.get("reference_label_normalized")),
                    "reference_source": clean_text(ref.get("reference_source")),
                    "reference_date": clean_text(ref.get("reference_date")),
                    "reference_latitude": None,
                    "reference_longitude": None,
                    "validation_source_independent": bool(ref.get("validation_source_independent")),
                    "validation_match_status": INVALID_REFERENCE,
                    "match_distance_km": np.nan,
                    "match_time_delta_hours": np.nan,
                    "candidate_match_count": 0,
                    "source_intelligence_candidate": None,
                    "evidence_strength": None,
                    "industrial_evidence_score": np.nan,
                }
            )
            continue

        idxs = tree.query_radius(np.radians([[float(rlat), float(rlon)]]), r=radius_rad)[0]
        if len(idxs) == 0:
            stats["no_event_match"] += 1
            rows.append(
                {
                    "validation_id": vid,
                    "event_id": None,
                    "reference_label_raw": clean_text(ref.get("reference_label_raw")),
                    "reference_label_normalized": clean_text(ref.get("reference_label_normalized")),
                    "reference_source": clean_text(ref.get("reference_source")),
                    "reference_date": clean_text(ref.get("reference_date")),
                    "reference_latitude": float(rlat),
                    "reference_longitude": float(rlon),
                    "validation_source_independent": bool(ref.get("validation_source_independent")),
                    "validation_match_status": NO_EVENT_MATCH,
                    "match_distance_km": np.nan,
                    "match_time_delta_hours": np.nan,
                    "candidate_match_count": 0,
                    "source_intelligence_candidate": None,
                    "evidence_strength": None,
                    "industrial_evidence_score": np.nan,
                }
            )
            continue

        # Temporal filter
        ref_time = pd.to_datetime(ref.get("reference_date"), utc=True, errors="coerce")
        survivors: list[tuple[int, float, float]] = []
        for idx in idxs:
            # haversine distance
            dlat = np.radians(lat[idx] - float(rlat))
            dlon = np.radians(lon[idx] - float(rlon))
            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(np.radians(float(rlat)))
                * np.cos(np.radians(lat[idx]))
                * np.sin(dlon / 2) ** 2
            )
            dist_km = float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))
            time_delta = np.nan
            if pd.notna(ref_time):
                start = ev.at[idx, "_start"]
                end = ev.at[idx, "_end"]
                if pd.isna(start) and pd.isna(end):
                    continue
                # distance to interval
                if pd.notna(start) and pd.notna(end) and start <= ref_time <= end:
                    time_delta = 0.0
                else:
                    candidates_t = []
                    if pd.notna(start):
                        candidates_t.append(abs((ref_time - start).total_seconds()) / 3600.0)
                    if pd.notna(end):
                        candidates_t.append(abs((ref_time - end).total_seconds()) / 3600.0)
                    time_delta = min(candidates_t) if candidates_t else np.nan
                    if time_delta > config.temporal_tolerance_hours:
                        continue
            survivors.append((int(idx), dist_km, float(time_delta) if pd.notna(time_delta) else np.nan))

        if not survivors:
            stats["no_event_match"] += 1
            status = NO_EVENT_MATCH
            chosen = None
        else:
            survivors.sort(
                key=lambda t: (
                    t[1],
                    t[2] if np.isfinite(t[2]) else 1e18,
                    str(ev.at[t[0], "event_id"]),
                )
            )
            if len(survivors) == 1:
                status = MATCHED
                chosen = survivors[0]
                stats["matched"] += 1
            else:
                # Ambiguous if top two are near-ties
                d0, d1 = survivors[0][1], survivors[1][1]
                if abs(d0 - d1) <= config.ambiguity_distance_tolerance_km:
                    status = MULTIPLE_POSSIBLE_MATCHES
                    chosen = None
                    stats["multiple_possible_matches"] += 1
                else:
                    status = MATCHED
                    chosen = survivors[0]
                    stats["matched"] += 1

        event_id = None
        cand = None
        strength = None
        score = np.nan
        dist = np.nan
        tdelta = np.nan
        if chosen is not None:
            idx, dist, tdelta = chosen
            event_id = str(ev.at[idx, "event_id"])
            if "source_intelligence_candidate" in candidate_cols:
                cand = clean_text(ev.at[idx, "source_intelligence_candidate"])
            if "evidence_strength" in candidate_cols:
                strength = clean_text(ev.at[idx, "evidence_strength"])
            if "industrial_evidence_score" in candidate_cols:
                score = pd.to_numeric(ev.at[idx, "industrial_evidence_score"], errors="coerce")

        rows.append(
            {
                "validation_id": vid,
                "event_id": event_id,
                "reference_label_raw": clean_text(ref.get("reference_label_raw")),
                "reference_label_normalized": clean_text(ref.get("reference_label_normalized")),
                "reference_source": clean_text(ref.get("reference_source")),
                "reference_date": clean_text(ref.get("reference_date")),
                "reference_latitude": float(rlat),
                "reference_longitude": float(rlon),
                "validation_source_independent": bool(ref.get("validation_source_independent")),
                "validation_match_status": status,
                "match_distance_km": dist,
                "match_time_delta_hours": tdelta,
                "candidate_match_count": int(len(survivors)),
                "source_intelligence_candidate": cand,
                "evidence_strength": strength,
                "industrial_evidence_score": score,
            }
        )

    matches = pd.DataFrame(rows).reindex(columns=list(MATCH_OUTPUT_COLUMNS))
    matches = matches.sort_values(["validation_id", "event_id"], kind="mergesort").reset_index(drop=True)
    return matches, stats
