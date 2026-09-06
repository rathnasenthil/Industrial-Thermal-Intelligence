"""Facility REST endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.events import PaginatedEvents
from app.schemas.facilities import FacilityDetail, PaginatedFacilities
from app.services import facilities as facilities_service

router = APIRouter(prefix="/facilities", tags=["facilities"])


@router.get("", response_model=PaginatedFacilities)
def list_facilities(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    facility_type: Optional[str] = None,
    search: Optional[str] = Query(None, description="Case-insensitive name/id search"),
    bbox: Optional[str] = Query(
        None,
        description="Bounding box min_lon,min_lat,max_lon,max_lat",
    ),
    db: Session = Depends(get_db),
) -> PaginatedFacilities:
    try:
        return facilities_service.list_facilities(
            db,
            page=page,
            page_size=page_size,
            facility_type=facility_type,
            search=search,
            bbox=bbox,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{facility_id}", response_model=FacilityDetail)
def get_facility(facility_id: str, db: Session = Depends(get_db)) -> FacilityDetail:
    detail = facilities_service.get_facility(db, facility_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Facility not found: {facility_id}")
    return detail


@router.get("/{facility_id}/history", response_model=PaginatedEvents)
def get_facility_history(
    facility_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    priority: Optional[str] = Query(
        None,
        description="Filter by investigation_priority",
    ),
    db: Session = Depends(get_db),
) -> PaginatedEvents:
    history = facilities_service.get_facility_history(
        db,
        facility_id,
        page=page,
        page_size=page_size,
        date_from=date_from,
        date_to=date_to,
        priority=priority,
    )
    if history is None:
        raise HTTPException(status_code=404, detail=f"Facility not found: {facility_id}")
    return history
