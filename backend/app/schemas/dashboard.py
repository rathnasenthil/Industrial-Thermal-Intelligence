"""Dashboard aggregate statistics schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class DashboardStatistics(BaseModel):
    """
    Live aggregates from PostgreSQL ΓÇö never hardcoded Stage VI counts.

    high_priority_count / critical_count refer to investigation_priority
    values, not emergency dispatch alerts or confirmed fires.
    """

    total_events: int
    total_facilities: int
    priority_distribution: dict[str, int] = Field(default_factory=dict)
    industrial_context_distribution: dict[str, int] = Field(default_factory=dict)
    persistence_distribution: dict[str, int] = Field(default_factory=dict)
    thermal_severity_distribution: dict[str, int] = Field(default_factory=dict)
    anomaly_distribution: dict[str, int] = Field(default_factory=dict)
    facility_type_distribution: dict[str, int] = Field(default_factory=dict)
    events_with_facility_association: int
    events_without_facility_association: int
    high_priority_count: int
    critical_count: int
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    semantics_note: str = (
        "Statistics describe the ingested Stage VI investigation dataset. "
        "risk_score / priority are decision-support fields, not fire probabilities "
        "or emergency dispatch status. Stage V produced no validated performance claim."
    )
