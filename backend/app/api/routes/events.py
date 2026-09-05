"""Thermal event REST endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.events import (
    EventDetail,
    EventEvidence,
    EventTimeline,
    PaginatedAlerts,
    PaginatedEvents,
)
from app.services import events as events_service

router = APIRouter(prefix="/events", tags=["events"])


@router.get("", response_model=PaginatedEvents)
def list_events(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    priority: Optional[str] = Query(
        None,
        description="Filter by investigation_priority (alias: priority)",
    ),
    industrial_context: Optional[str] = None,
    facility_type: Optional[str] = None,
    persistence_class: Optional[str] = Query(
        None,
        description="Filter by persistence_label (alias: persistence_class)",
    ),
    anomaly_status: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    min_risk_score: Optional[float] = Query(None, ge=0, le=100),
    max_risk_score: Optional[float] = Query(None, ge=0, le=100),
    bbox: Optional[str] = Query(
        None,
        description="Bounding box min_lon,min_lat,max_lon,max_lat (PostGIS ST_Intersects)",
    ),
    db: Session = Depends(get_db),
) -> PaginatedEvents:
    try:
        return events_service.list_events(
            db,
            page=page,
            page_size=page_size,
            priority=priority,
            industrial_context=industrial_context,
            facility_type=facility_type,
            persistence_class=persistence_class,
            anomaly_status=anomaly_status,
            date_from=date_from,
            date_to=date_to,
            min_risk_score=min_risk_score,
            max_risk_score=max_risk_score,
            bbox=bbox,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{event_id}", response_model=EventDetail)
def get_event(event_id: str, db: Session = Depends(get_db)) -> EventDetail:
    detail = events_service.get_event(db, event_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Event not found: {event_id}")
    return detail


@router.get("/{event_id}/evidence", response_model=EventEvidence)
def get_event_evidence(event_id: str, db: Session = Depends(get_db)) -> EventEvidence:
    evidence = events_service.get_event_evidence(db, event_id)
    if evidence is None:
        raise HTTPException(status_code=404, detail=f"Event not found: {event_id}")
    return evidence


@router.get("/{event_id}/timeline", response_model=EventTimeline)
def get_event_timeline(event_id: str, db: Session = Depends(get_db)) -> EventTimeline:
    timeline = events_service.get_event_timeline(db, event_id)
    if timeline is None:
        raise HTTPException(status_code=404, detail=f"Event not found: {event_id}")
    return timeline
