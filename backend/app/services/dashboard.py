"""Dashboard aggregate statistics from PostgreSQL."""

from __future__ import annotations

from sqlalchemy import func, literal, select, union_all
from sqlalchemy.orm import Session

from app.models.facility import Facility
from app.models.thermal_event import ThermalEvent
from app.schemas.dashboard import DashboardStatistics


def _distributions(db: Session) -> dict[str, dict[str, int]]:
    """
    Fetch all dashboard distributions through one database round-trip.

    Each distribution still uses PostgreSQL GROUP BY, but combining them
    into one UNION ALL statement avoids six separate database round-trips.
    """

    queries = [
        (
            "priority",
            ThermalEvent.investigation_priority,
        ),
        (
            "industrial_context",
            ThermalEvent.industrial_context,
        ),
        (
            "persistence",
            ThermalEvent.persistence_label,
        ),
        (
            "thermal_severity",
            ThermalEvent.thermal_severity_band,
        ),
        (
            "anomaly",
            ThermalEvent.anomaly_status,
        ),
        (
            "facility_type",
            ThermalEvent.facility_type,
        ),
    ]

    statements = []

    for distribution_name, column in queries:
        statements.append(
            select(
                literal(distribution_name).label("distribution"),
                column.label("category"),
                func.count().label("count"),
            )
            .where(column.is_not(None))
            .group_by(column)
        )

    rows = db.execute(union_all(*statements)).all()

    result: dict[str, dict[str, int]] = {
        "priority": {},
        "industrial_context": {},
        "persistence": {},
        "thermal_severity": {},
        "anomaly": {},
        "facility_type": {},
    }

    for distribution, category, count in rows:
        result[str(distribution)][str(category)] = int(count)

    return result


def get_dashboard_statistics(db: Session) -> DashboardStatistics:
    """
    Return dashboard aggregates from PostgreSQL.

    Core event totals, priority counts, association counts, and date
    boundaries are calculated in one aggregate query. Distribution
    queries are consolidated into one database round-trip.
    """

    # ---------------------------------------------------------
    # Core event statistics
    # ---------------------------------------------------------
    event_stats = db.execute(
        select(
            func.count(ThermalEvent.event_id).label("total_events"),

            func.count(ThermalEvent.event_id)
            .filter(
                ThermalEvent.facility_id.is_not(None),
                ThermalEvent.facility_association_method.is_not(None),
                ThermalEvent.facility_association_method
                != "NO_FACILITY_ASSOCIATION",
            )
            .label("with_assoc"),

            func.count(ThermalEvent.event_id)
            .filter(
                (ThermalEvent.facility_id.is_(None))
                | (
                    ThermalEvent.facility_association_method
                    == "NO_FACILITY_ASSOCIATION"
                )
            )
            .label("without_assoc"),

            func.count(ThermalEvent.event_id)
            .filter(
                ThermalEvent.investigation_priority == "HIGH"
            )
            .label("high_priority"),

            func.count(ThermalEvent.event_id)
            .filter(
                ThermalEvent.investigation_priority == "CRITICAL"
            )
            .label("critical"),

            func.min(ThermalEvent.event_start).label("date_start"),
            func.max(ThermalEvent.event_end).label("date_end"),
        )
    ).one()

    # ---------------------------------------------------------
    # Facility universe count
    # ---------------------------------------------------------
    total_facilities = db.scalar(
        select(func.count(Facility.facility_id))
    ) or 0

    # ---------------------------------------------------------
    # All distributions
    # ---------------------------------------------------------
    distributions = _distributions(db)

    # ---------------------------------------------------------
    # API response
    # ---------------------------------------------------------
    return DashboardStatistics(
        total_events=int(event_stats.total_events or 0),
        total_facilities=int(total_facilities),

        priority_distribution=distributions["priority"],

        industrial_context_distribution=
            distributions["industrial_context"],

        persistence_distribution=
            distributions["persistence"],

        thermal_severity_distribution=
            distributions["thermal_severity"],

        anomaly_distribution=
            distributions["anomaly"],

        facility_type_distribution=
            distributions["facility_type"],

        events_with_facility_association=
            int(event_stats.with_assoc or 0),

        events_without_facility_association=
            int(event_stats.without_assoc or 0),

        high_priority_count=
            int(event_stats.high_priority or 0),

        critical_count=
            int(event_stats.critical or 0),

        date_range_start=event_stats.date_start,
        date_range_end=event_stats.date_end,
    )