"""
Investigation-priority alert view.

Returns HIGH/CRITICAL investigation_priority events. This is not an emergency
dispatch system and does not generate push notifications.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.events import PaginatedAlerts
from app.services import events as events_service

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get(
    "",
    response_model=PaginatedAlerts,
    summary="Investigation-priority events (HIGH/CRITICAL)",
    description=(
        "Filtered view of events with investigation_priority in {HIGH, CRITICAL}. "
        "This is an investigation-priority queue for analysts — not an emergency "
        "dispatch or confirmed-fire alert engine. risk_score remains a "
        "decision-support score, not a probability."
    ),
)
def list_alerts(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    industrial_context: Optional[str] = None,
    facility_type: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    min_risk_score: Optional[float] = Query(None, ge=0, le=100),
    max_risk_score: Optional[float] = Query(None, ge=0, le=100),
    bbox: Optional[str] = Query(None),
    db: Session = Depends(get_db),
) -> PaginatedAlerts:
    try:
        return events_service.list_alerts(
            db,
            page=page,
            page_size=page_size,
            industrial_context=industrial_context,
            facility_type=facility_type,
            date_from=date_from,
            date_to=date_to,
            min_risk_score=min_risk_score,
            max_risk_score=max_risk_score,
            bbox=bbox,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
