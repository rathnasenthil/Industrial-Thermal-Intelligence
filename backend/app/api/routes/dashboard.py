"""Dashboard statistics endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.dashboard import DashboardStatistics
from app.services import dashboard as dashboard_service

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/statistics", response_model=DashboardStatistics)
def get_statistics(db: Session = Depends(get_db)) -> DashboardStatistics:
    """Aggregate statistics computed live from PostgreSQL."""
    return dashboard_service.get_dashboard_statistics(db)
