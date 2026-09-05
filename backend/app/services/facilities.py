"""Facility query helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import Session

from app.models.facility import Facility
from app.models.thermal_event import ThermalEvent
from app.schemas.common import pagination_totals, parse_bbox, point_from_lon_lat
from app.schemas.facilities import (
    FacilityDetail,
    FacilitySummary,
    FacilityThermalSummary,
    PaginatedFacilities,
)
from app.schemas.events import PaginatedEvents
from app.services.events import list_events


def _apply_facility_filters(
    stmt: Select,
    *,
    facility_type: Optional[str] = None,
    search: Optional[str] = None,
    bbox: Optional[str] = None,
) -> Select:
    if facility_type:
        stmt = stmt.where(Facility.facility_type == facility_type)
    if search:
        pattern = f"%{search.strip()}%"
        stmt = stmt.where(
            or_(
                Facility.facility_name.ilike(pattern),
                Facility.facility_id.ilike(pattern),
            )
        )
    if bbox:
        box = parse_bbox(bbox)
        envelope = func.ST_MakeEnvelope(
            box.min_lon, box.min_lat, box.max_lon, box.max_lat, 4326
        )
        stmt = stmt.where(func.ST_Intersects(Facility.geometry, envelope))
    return stmt


def facility_to_summary(facility: Facility) -> FacilitySummary:
    return FacilitySummary(
        facility_id=facility.facility_id,
        facility_name=facility.facility_name,
        facility_type=facility.facility_type,
        latitude=facility.latitude,
        longitude=facility.longitude,
        geometry=point_from_lon_lat(facility.longitude, facility.latitude),
        osm_id=facility.osm_id,
        osm_type=facility.osm_type,
        confidence=facility.confidence,
    )


def list_facilities(
    db: Session,
    *,
    page: int = 1,
    page_size: int = 50,
    facility_type: Optional[str] = None,
    search: Optional[str] = None,
    bbox: Optional[str] = None,
) -> PaginatedFacilities:
    page = max(page, 1)
    page_size = min(max(page_size, 1), 500)

    base = select(Facility)
    base = _apply_facility_filters(
        base, facility_type=facility_type, search=search, bbox=bbox
    )
    total = int(db.scalar(select(func.count()).select_from(base.subquery())) or 0)
    rows = list(
        db.scalars(
            base.order_by(Facility.facility_name.asc().nullslast(), Facility.facility_id)
            .offset((page - 1) * page_size)
            .limit(page_size)
        ).all()
    )
    meta = pagination_totals(total, page, page_size)
    return PaginatedFacilities(
        items=[facility_to_summary(r) for r in rows],
        total=meta.total,
        page=meta.page,
        page_size=meta.page_size,
        total_pages=meta.total_pages,
    )


def _facility_thermal_summary(db: Session, facility_id: str) -> FacilityThermalSummary:
    count = int(
        db.scalar(
            select(func.count()).where(ThermalEvent.facility_id == facility_id)
        )
        or 0
    )
    high = int(
        db.scalar(
            select(func.count()).where(
                ThermalEvent.facility_id == facility_id,
                ThermalEvent.investigation_priority == "HIGH",
            )
        )
        or 0
    )
    critical = int(
        db.scalar(
            select(func.count()).where(
                ThermalEvent.facility_id == facility_id,
                ThermalEvent.investigation_priority == "CRITICAL",
            )
        )
        or 0
    )
    max_risk = db.scalar(
        select(func.max(ThermalEvent.risk_score)).where(
            ThermalEvent.facility_id == facility_id
        )
    )
    latest = db.scalar(
        select(func.max(ThermalEvent.event_start)).where(
            ThermalEvent.facility_id == facility_id
        )
    )
    return FacilityThermalSummary(
        associated_event_count=count,
        high_priority_count=high,
        critical_count=critical,
        max_risk_score=float(max_risk) if max_risk is not None else None,
        latest_event_start=latest.isoformat() if latest is not None else None,
    )


def get_facility(db: Session, facility_id: str) -> Optional[FacilityDetail]:
    facility = db.scalar(select(Facility).where(Facility.facility_id == facility_id))
    if facility is None:
        return None
    summary = facility_to_summary(facility)
    return FacilityDetail(
        **summary.model_dump(),
        industrial_subtype=facility.industrial_subtype,
        operator=facility.operator,
        landuse=facility.landuse,
        power_type=facility.power_type,
        man_made_type=facility.man_made_type,
        geometry_type=facility.geometry_type,
        geometry_wkt=facility.geometry_wkt,
        osm_tags=facility.osm_tags,
        source=facility.source,
        source_version=facility.source_version,
        thermal_summary=_facility_thermal_summary(db, facility_id),
    )


def get_facility_history(
    db: Session,
    facility_id: str,
    *,
    page: int = 1,
    page_size: int = 50,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    priority: Optional[str] = None,
) -> Optional[PaginatedEvents]:
    exists = db.scalar(select(Facility.id).where(Facility.facility_id == facility_id))
    if exists is None:
        return None
    return list_events(
        db,
        page=page,
        page_size=page_size,
        facility_id=facility_id,
        date_from=date_from,
        date_to=date_to,
        priority=priority,
    )
