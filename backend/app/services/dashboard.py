"""Dashboard aggregate statistics from PostgreSQL."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.facility import Facility
from app.models.thermal_event import ThermalEvent
from app.schemas.dashboard import DashboardStatistics


def _distribution(db: Session, column) -> dict[str, int]:
    rows = db.execute(
        select(column, func.count())
        .where(column.is_not(None))
        .group_by(column)
        .order_by(func.count().desc())
    ).all()
    return {str(key): int(count) for key, count in rows}


def get_dashboard_statistics(db: Session) -> DashboardStatistics:
    total_events = int(db.scalar(select(func.count()).select_from(ThermalEvent)) or 0)
    total_facilities = int(db.scalar(select(func.count()).select_from(Facility)) or 0)

    with_assoc = int(
        db.scalar(
            select(func.count()).where(
                ThermalEvent.facility_id.is_not(None),
                ThermalEvent.facility_association_method.is_not(None),
                ThermalEvent.facility_association_method
                != "NO_FACILITY_ASSOCIATION",
            )
        )
        or 0
    )
    without_assoc = int(
        db.scalar(
            select(func.count()).where(
                (ThermalEvent.facility_id.is_(None))
                | (ThermalEvent.facility_association_method == "NO_FACILITY_ASSOCIATION")
            )
        )
        or 0
    )

    high_priority = int(
        db.scalar(
            select(func.count()).where(ThermalEvent.investigation_priority == "HIGH")
        )
        or 0
    )
    critical = int(
        db.scalar(
            select(func.count()).where(
                ThermalEvent.investigation_priority == "CRITICAL"
            )
        )
        or 0
    )

    date_start = db.scalar(select(func.min(ThermalEvent.event_start)))
    date_end = db.scalar(select(func.max(ThermalEvent.event_end)))

    # Facility-type distribution prefers event-side attributed types for dashboard
    # context of observed thermal activity; facility universe distribution is
    # available separately via facilities API filters.
    return DashboardStatistics(
        total_events=total_events,
        total_facilities=total_facilities,
        priority_distribution=_distribution(db, ThermalEvent.investigation_priority),
        industrial_context_distribution=_distribution(
            db, ThermalEvent.industrial_context
        ),
        persistence_distribution=_distribution(db, ThermalEvent.persistence_label),
        thermal_severity_distribution=_distribution(
            db, ThermalEvent.thermal_severity_band
        ),
        anomaly_distribution=_distribution(db, ThermalEvent.anomaly_status),
        facility_type_distribution=_distribution(db, ThermalEvent.facility_type),
        events_with_facility_association=with_assoc,
        events_without_facility_association=without_assoc,
        high_priority_count=high_priority,
        critical_count=critical,
        date_range_start=date_start,
        date_range_end=date_end,
    )
