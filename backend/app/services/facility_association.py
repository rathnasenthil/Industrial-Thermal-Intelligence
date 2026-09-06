"""
Backend Phase 5: incremental Stage I.2 facility association.

Loads one thermal event + spatially narrowed facilities (using full
``geometry_wkt``, not the POINT-only PostGIS column), calls AIML
``realtime.facility_association``, and upserts:

- ``event_facility_candidates`` for that event only
- selected association columns on ``thermal_events``

Facility association is spatial attribution only — not source
classification or industrial-fire confirmation.

Does **not** call ``run_facility_association()`` over historical events.
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models.event_detection import EventDetection
from app.models.event_facility_candidate import EventFacilityCandidate
from app.models.facility import Facility
from app.models.firms_observation import FirmsObservation
from app.models.thermal_event import ThermalEvent

logger = logging.getLogger(__name__)

_AIML_ROOT = Path(__file__).resolve().parents[3] / "aiml"
if str(_AIML_ROOT) not in sys.path:
    sys.path.insert(0, str(_AIML_ROOT))

from realtime.facility_association import (  # noqa: E402
    AssociationResult,
    FacilityRecord,
    build_event_wkt_from_coordinates,
    process_event_facility_association,
)
from src.infrastructure.association_config import AssociationConfig, DEFAULT_CONFIG  # noqa: E402

# Extra margin beyond association_radius so large OSM polygons whose
# centroids lie slightly outside the radius are still considered by AIML
# (exact I.2 filtering remains in AIML). Does not change semantics.
_FACILITY_PREFILTER_MARGIN_KM = 15.0


@dataclass
class FacilityAssociationStats:
    events_processed: int = 0
    candidates_written: int = 0
    associations_with_facility: int = 0
    associations_none: int = 0
    associations_ambiguous: int = 0
    event_ids: list[str] = field(default_factory=list)
    by_event: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_event_detection_coordinates(
    session: Session, event_id: str
) -> tuple[list[float], list[float]]:
    rows = session.execute(
        select(FirmsObservation.latitude, FirmsObservation.longitude)
        .join(
            EventDetection,
            EventDetection.observation_hash == FirmsObservation.observation_hash,
        )
        .where(EventDetection.event_id == event_id)
        .order_by(FirmsObservation.acq_datetime.asc().nullslast(), FirmsObservation.id.asc())
    ).all()
    lats: list[float] = []
    lons: list[float] = []
    for lat, lon in rows:
        if lat is None or lon is None:
            continue
        lats.append(float(lat))
        lons.append(float(lon))
    return lats, lons


def ensure_event_geometry_wkt(session: Session, event: ThermalEvent) -> tuple[Optional[str], Optional[str]]:
    """
    Ensure centroid_wkt / footprint_wkt match Stage G (from detections).

    Realtime Phase 3 may leave footprint_wkt null; recompute from member
    detections when needed so I.2 buffer/join semantics stay correct.
    """
    lats, lons = load_event_detection_coordinates(session, event.event_id)
    if lats and lons:
        centroid_wkt, footprint_wkt = build_event_wkt_from_coordinates(lats, lons)
        event.centroid_wkt = centroid_wkt
        event.footprint_wkt = footprint_wkt
        # Keep numeric centroid aligned with WKT.
        event.centroid_latitude = float(np.mean(lats))
        event.centroid_longitude = float(np.mean(lons))
        event.min_latitude = float(min(lats))
        event.max_latitude = float(max(lats))
        event.min_longitude = float(min(lons))
        event.max_longitude = float(max(lons))
        event.updated_at = datetime.now(timezone.utc)
        session.flush()
        return centroid_wkt, footprint_wkt

    if event.centroid_wkt:
        footprint = event.footprint_wkt or event.centroid_wkt
        if event.footprint_wkt is None:
            event.footprint_wkt = footprint
            session.flush()
        return event.centroid_wkt, footprint

    if event.centroid_latitude is not None and event.centroid_longitude is not None:
        lon = float(event.centroid_longitude)
        lat = float(event.centroid_latitude)
        centroid_wkt = f"POINT ({lon} {lat})"
        event.centroid_wkt = centroid_wkt
        event.footprint_wkt = centroid_wkt
        session.flush()
        return centroid_wkt, centroid_wkt

    return None, None


def _degrees_for_km(km: float, latitude: float) -> tuple[float, float]:
    """Rough degree deltas for a lat/lon bbox prefilter (India latitudes)."""
    lat_delta = km / 111.0
    cos_lat = max(math.cos(math.radians(latitude)), 0.2)
    lon_delta = km / (111.0 * cos_lat)
    return lat_delta, lon_delta


def fetch_nearby_facilities(
    session: Session,
    *,
    latitude: float,
    longitude: float,
    radius_km: float,
    margin_km: float = _FACILITY_PREFILTER_MARGIN_KM,
) -> list[FacilityRecord]:
    """
    Spatially narrow facilities for AIML I.2.

    Uses lat/lon bbox around the event plus margin. Exact candidate
    retention uses full ``geometry_wkt`` inside AIML (polygon-aware).
    """
    search_km = radius_km + margin_km
    lat_d, lon_d = _degrees_for_km(search_km, latitude)
    rows = session.scalars(
        select(Facility).where(
            Facility.latitude.is_not(None),
            Facility.longitude.is_not(None),
            Facility.geometry_wkt.is_not(None),
            Facility.latitude >= latitude - lat_d,
            Facility.latitude <= latitude + lat_d,
            Facility.longitude >= longitude - lon_d,
            Facility.longitude <= longitude + lon_d,
        )
    ).all()
    out: list[FacilityRecord] = []
    for f in rows:
        if not f.geometry_wkt or not f.facility_id:
            continue
        out.append(
            FacilityRecord(
                facility_id=f.facility_id,
                facility_name=f.facility_name,
                facility_type=f.facility_type,
                geometry_type=f.geometry_type,
                geometry_wkt=f.geometry_wkt,
            )
        )
    return out


def apply_association_to_event(event: ThermalEvent, result: AssociationResult) -> None:
    event.facility_id = result.facility_id
    event.facility_name = result.facility_name
    event.facility_type = result.facility_type
    event.facility_association_method = result.facility_association_method
    event.facility_attribution_confidence = result.facility_attribution_confidence
    event.facility_distance_km = result.facility_distance_km
    event.candidate_facility_count = result.candidate_facility_count
    if result.centroid_wkt:
        event.centroid_wkt = result.centroid_wkt
    if result.footprint_wkt:
        event.footprint_wkt = result.footprint_wkt
    event.updated_at = datetime.now(timezone.utc)


def replace_event_candidates(
    session: Session,
    event_id: str,
    result: AssociationResult,
) -> int:
    """
    Replace candidate rows for one event (idempotent for identical results).

    Deletes only ``event_facility_candidates`` for ``event_id``, never
    historical candidates for other events.
    """
    session.execute(
        delete(EventFacilityCandidate).where(EventFacilityCandidate.event_id == event_id)
    )
    for c in result.candidates:
        session.add(
            EventFacilityCandidate(
                event_id=event_id,
                facility_id=c.facility_id,
                facility_name=c.facility_name,
                facility_type=c.facility_type,
                spatial_relation=c.spatial_relation,
                distance_km=c.distance_km,
                candidate_rank=c.candidate_rank,
                candidate_score=c.candidate_score,
            )
        )
    session.flush()
    return len(result.candidates)


def refresh_event_facility_association(
    session: Session,
    event_id: str,
    *,
    config: Optional[AssociationConfig] = None,
) -> AssociationResult:
    """
    Phase 5: I.2 for one event only.

    Does not modify other events' candidates or historical associations
    except the single ``event_id`` row being refreshed.
    """
    cfg = config or DEFAULT_CONFIG
    event = session.scalar(select(ThermalEvent).where(ThermalEvent.event_id == event_id))
    if event is None:
        raise RuntimeError(f"ThermalEvent missing for facility association: {event_id}")

    centroid_wkt, footprint_wkt = ensure_event_geometry_wkt(session, event)
    if not centroid_wkt:
        result = process_event_facility_association(
            event_id,
            centroid_wkt=None,
            footprint_wkt=None,
            facilities=[],
            config=cfg,
        )
        apply_association_to_event(event, result)
        replace_event_candidates(session, event_id, result)
        return result

    lat = float(event.centroid_latitude) if event.centroid_latitude is not None else None
    lon = float(event.centroid_longitude) if event.centroid_longitude is not None else None
    if lat is None or lon is None:
        # Parse from WKT as last resort — AIML still needs facilities list.
        facilities: list[FacilityRecord] = []
    else:
        facilities = fetch_nearby_facilities(
            session,
            latitude=lat,
            longitude=lon,
            radius_km=cfg.association_radius_km,
        )

    result = process_event_facility_association(
        event_id,
        centroid_wkt=centroid_wkt,
        footprint_wkt=footprint_wkt,
        facilities=facilities,
        config=cfg,
    )
    apply_association_to_event(event, result)
    replace_event_candidates(session, event_id, result)
    session.flush()
    return result


def associate_events(
    session: Session,
    event_ids: Sequence[str],
    *,
    config: Optional[AssociationConfig] = None,
    commit: bool = False,
) -> FacilityAssociationStats:
    """Run Phase 5 for each distinct event_id (deduplicated)."""
    cfg = config or DEFAULT_CONFIG
    stats = FacilityAssociationStats()
    for eid in sorted(set(event_ids)):
        result = refresh_event_facility_association(session, eid, config=cfg)
        stats.events_processed += 1
        stats.event_ids.append(eid)
        stats.candidates_written += len(result.candidates)
        method = result.facility_association_method
        if method == "AMBIGUOUS":
            stats.associations_ambiguous += 1
        elif method == "NO_FACILITY_ASSOCIATION":
            stats.associations_none += 1
        elif result.facility_id:
            stats.associations_with_facility += 1
        else:
            stats.associations_none += 1
        stats.by_event[eid] = {
            "facility_id": result.facility_id,
            "facility_association_method": result.facility_association_method,
            "facility_attribution_confidence": result.facility_attribution_confidence,
            "facility_distance_km": result.facility_distance_km,
            "candidate_facility_count": result.candidate_facility_count,
            "candidates": [c.to_dict() for c in result.candidates],
        }
    if commit:
        session.commit()
    else:
        session.flush()
    logger.info(
        "Phase 5 facility association: events=%s with_facility=%s none=%s ambiguous=%s",
        stats.events_processed,
        stats.associations_with_facility,
        stats.associations_none,
        stats.associations_ambiguous,
    )
    return stats
