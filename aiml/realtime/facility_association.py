"""
Incremental Stage I.2 facility association (AIML realtime adapter).

Facility association is **spatial attribution only**.

It does NOT mean:
- source classification
- proof of industrial activity
- confirmation of an industrial fire
- that the facility caused the thermal event

Why realtime I.2 cannot replay the full batch pipeline
------------------------------------------------------
Batch ``run_facility_association()`` joins every Stage G/G.1 event against
the full facility layer (~180k × ~113k). On each NRT poll only the
affected event's geometry may change. Re-running the batch entrypoint
would re-touch unrelated historical rows and is unnecessary.

Instead: for **one** event, reuse the same batch primitives:

1. ``build_event_geometries`` / ``find_candidate_pairs``
2. ``rank_candidates``
3. ``select_association``

with the same ``AssociationConfig`` defaults (5 km radius, 0.5 km
ambiguity tolerance, max 10 candidates in the detail list).

Why ranking is not "nearest facility"
-------------------------------------
Batch I.2 ranks by spatial-relation tier (WITHIN > INTERSECTS > NEAR),
then distance, then geometry quality, then deterministic lexical
tie-breaks. Ambiguous top-two candidates in the same tier within the
ambiguity tolerance yield AMBIGUOUS with **no** selected facility —
explicitly rejecting blind nearest-neighbor selection.

Event movement
--------------
When additional detections move the centroid/footprint, association must
be recomputed for that event. The first association is not locked.

Null / invalid geometry
-----------------------
Events or facilities without parseable geometry cannot participate;
the event receives ``NO_FACILITY_ASSOCIATION`` rather than a fabricated
match.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isnan
from typing import Any, Mapping, Optional, Sequence

import geopandas as gpd
import pandas as pd
from shapely import wkt as shapely_wkt

from src.event_formation.geometry import compute_event_geometry
from src.infrastructure.association_config import AssociationConfig, DEFAULT_CONFIG
from src.infrastructure.association_geometry import (
    FACILITY_COLUMNS,
    build_event_geometries,
    find_candidate_pairs,
)
from src.infrastructure.facility_association import (
    CANDIDATES_OUTPUT_COLUMNS,
    MAIN_OUTPUT_COLUMNS,
    NO_FACILITY_ASSOCIATION,
    CONFIDENCE_NONE,
    rank_candidates,
    select_association,
)
from src.infrastructure.facility_schema import SUPPORTED_GEOMETRY_TYPES

# Test hook: realtime path must not invoke the full-batch orchestrator.
_BATCH_PIPELINE_INVOCATIONS = 0


@dataclass(frozen=True)
class FacilityRecord:
    """Plain facility inputs for realtime I.2 (no ORM)."""

    facility_id: str
    facility_name: Optional[str]
    facility_type: Optional[str]
    geometry_type: Optional[str]
    geometry_wkt: str


@dataclass(frozen=True)
class CandidateAssociation:
    event_id: str
    facility_id: str
    facility_name: Optional[str]
    facility_type: Optional[str]
    spatial_relation: str
    distance_km: float
    candidate_rank: int
    candidate_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AssociationResult:
    """Framework-independent I.2 result for one thermal event."""

    event_id: str
    facility_id: Optional[str]
    facility_name: Optional[str]
    facility_type: Optional[str]
    facility_association_method: str
    facility_attribution_confidence: str
    facility_distance_km: Optional[float]
    candidate_facility_count: int
    candidate_facility_ids: str
    candidates: tuple[CandidateAssociation, ...] = field(default_factory=tuple)
    centroid_wkt: Optional[str] = None
    footprint_wkt: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["candidates"] = [c.to_dict() for c in self.candidates]
        return out


def batch_pipeline_invocation_count() -> int:
    return _BATCH_PIPELINE_INVOCATIONS


def _nan_to_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if isnan(f):
        return None
    return f


def build_event_wkt_from_coordinates(
    latitudes: Sequence[float],
    longitudes: Sequence[float],
) -> tuple[str, str]:
    """
    Stage G centroid + convex-hull footprint WKT from member detections.

    Same formulas as ``compute_event_geometry``.
    """
    import numpy as np

    if not latitudes or not longitudes:
        raise ValueError("cannot build event geometry from empty coordinates")
    if len(latitudes) != len(longitudes):
        raise ValueError("latitude/longitude length mismatch")
    geom = compute_event_geometry(
        np.asarray(latitudes, dtype=float),
        np.asarray(longitudes, dtype=float),
    )
    return geom.centroid_wkt, geom.footprint_wkt


def facilities_geodataframe_from_records(
    facilities: Sequence[FacilityRecord | Mapping[str, Any]],
) -> gpd.GeoDataFrame:
    """
    Build an EPSG:4326 GeoDataFrame matching Stage I.1 column expectations.

    Drops rows with missing/unsupported/unparseable geometry (same defensive
    policy as ``load_facilities_geodataframe``).
    """
    rows: list[dict[str, Any]] = []
    geoms = []
    for item in facilities:
        if isinstance(item, FacilityRecord):
            rec = asdict(item)
        else:
            rec = dict(item)
        wkt = rec.get("geometry_wkt")
        gtype = rec.get("geometry_type")
        if not isinstance(wkt, str) or not wkt.strip():
            continue
        if gtype not in SUPPORTED_GEOMETRY_TYPES:
            continue
        try:
            geom = shapely_wkt.loads(wkt)
        except Exception:
            continue
        if geom is None or geom.is_empty:
            continue
        rows.append(
            {
                "facility_id": str(rec["facility_id"]),
                "facility_name": rec.get("facility_name"),
                "facility_type": rec.get("facility_type"),
                "geometry_type": gtype,
            }
        )
        geoms.append(geom)

    if not rows:
        return gpd.GeoDataFrame(columns=[*FACILITY_COLUMNS, "geometry"], geometry="geometry", crs="EPSG:4326")

    return gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")


def _empty_association(event_id: str, *, centroid_wkt: Optional[str] = None, footprint_wkt: Optional[str] = None) -> AssociationResult:
    return AssociationResult(
        event_id=event_id,
        facility_id=None,
        facility_name=None,
        facility_type=None,
        facility_association_method=NO_FACILITY_ASSOCIATION,
        facility_attribution_confidence=CONFIDENCE_NONE,
        facility_distance_km=None,
        candidate_facility_count=0,
        candidate_facility_ids="",
        candidates=(),
        centroid_wkt=centroid_wkt,
        footprint_wkt=footprint_wkt,
    )


def process_event_facility_association(
    event_id: str,
    *,
    centroid_wkt: Optional[str],
    footprint_wkt: Optional[str],
    facilities: Sequence[FacilityRecord | Mapping[str, Any]],
    config: Optional[AssociationConfig] = None,
) -> AssociationResult:
    """
    Run Stage I.2 for **one** thermal event against a facility subset.

    The facility subset may be spatially pre-filtered by the backend as long
    as it is a *superset* of facilities that could match under
    ``association_radius_km`` (exact filtering remains here).

    Does **not** call ``run_facility_association()`` over all events.
    """
    cfg = config or DEFAULT_CONFIG

    if not centroid_wkt or not isinstance(centroid_wkt, str) or not centroid_wkt.strip():
        return _empty_association(event_id)
    if not footprint_wkt or not isinstance(footprint_wkt, str) or not footprint_wkt.strip():
        # Single-detection fallback: footprint == centroid point (Stage G).
        footprint_wkt = centroid_wkt

    try:
        shapely_wkt.loads(centroid_wkt)
        shapely_wkt.loads(footprint_wkt)
    except Exception:
        return _empty_association(event_id)

    events_df = pd.DataFrame(
        [
            {
                "event_id": event_id,
                "centroid_wkt": centroid_wkt,
                "footprint_wkt": footprint_wkt,
            }
        ]
    )
    try:
        events_gdf = build_event_geometries(events_df)
    except ValueError:
        return _empty_association(event_id)

    facilities_gdf = facilities_geodataframe_from_records(facilities)
    pairs_df = find_candidate_pairs(events_gdf, facilities_gdf, cfg.association_radius_km)
    ranked_df = rank_candidates(pairs_df)
    selection_df = select_association(events_df["event_id"], ranked_df, cfg)
    row = selection_df.iloc[0]

    candidates_df = (
        ranked_df[list(CANDIDATES_OUTPUT_COLUMNS)].copy()
        if not ranked_df.empty
        else pd.DataFrame(columns=list(CANDIDATES_OUTPUT_COLUMNS))
    )
    if cfg.max_candidates_per_event is not None and not candidates_df.empty:
        candidates_df = candidates_df.loc[
            candidates_df["candidate_rank"] <= cfg.max_candidates_per_event
        ]

    candidates = tuple(
        CandidateAssociation(
            event_id=str(r["event_id"]),
            facility_id=str(r["facility_id"]),
            facility_name=None if pd.isna(r["facility_name"]) else str(r["facility_name"]),
            facility_type=None if pd.isna(r["facility_type"]) else str(r["facility_type"]),
            spatial_relation=str(r["spatial_relation"]),
            distance_km=float(r["distance_km"]),
            candidate_rank=int(r["candidate_rank"]),
            candidate_score=float(r["candidate_score"]),
        )
        for _, r in candidates_df.iterrows()
    )

    return AssociationResult(
        event_id=event_id,
        facility_id=None if pd.isna(row["facility_id"]) else str(row["facility_id"]),
        facility_name=None if pd.isna(row["facility_name"]) else str(row["facility_name"]),
        facility_type=None if pd.isna(row["facility_type"]) else str(row["facility_type"]),
        facility_association_method=str(row["facility_association_method"]),
        facility_attribution_confidence=str(row["facility_attribution_confidence"]),
        facility_distance_km=_nan_to_none(row["facility_distance_km"]),
        candidate_facility_count=int(row["candidate_facility_count"]),
        candidate_facility_ids=str(row["candidate_facility_ids"] or ""),
        candidates=candidates,
        centroid_wkt=centroid_wkt,
        footprint_wkt=footprint_wkt,
    )


# Ensure MAIN_OUTPUT_COLUMNS stay aligned with batch (import side-effect for tests).
assert set(MAIN_OUTPUT_COLUMNS) >= {
    "facility_id",
    "facility_association_method",
    "facility_attribution_confidence",
    "facility_distance_km",
    "candidate_facility_count",
}
