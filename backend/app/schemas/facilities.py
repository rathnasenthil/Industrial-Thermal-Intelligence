"""Pydantic schemas for facility API responses."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import GeometryPoint, PaginatedResponse
from app.schemas.events import EventSummary


class FacilitySummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    facility_id: str
    facility_name: Optional[str] = None
    facility_type: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    geometry: Optional[GeometryPoint] = None
    osm_id: Optional[str] = None
    osm_type: Optional[str] = None
    confidence: Optional[str] = None


class FacilityThermalSummary(BaseModel):
    """Aggregate of associated Stage VI events (not a validated ML metric)."""

    associated_event_count: int = 0
    high_priority_count: int = 0
    critical_count: int = 0
    max_risk_score: Optional[float] = None
    latest_event_start: Optional[str] = None


class FacilityDetail(FacilitySummary):
    industrial_subtype: Optional[str] = None
    operator: Optional[str] = None
    landuse: Optional[str] = None
    power_type: Optional[str] = None
    man_made_type: Optional[str] = None
    geometry_type: Optional[str] = None
    geometry_wkt: Optional[str] = None
    osm_tags: Optional[dict] = None
    source: Optional[str] = None
    source_version: Optional[str] = None
    thermal_summary: FacilityThermalSummary = Field(default_factory=FacilityThermalSummary)
    semantics_note: str = (
        "Facility association with thermal events is spatial attribution, "
        "not source classification or confirmed industrial fire."
    )


PaginatedFacilities = PaginatedResponse[FacilitySummary]
PaginatedFacilityHistory = PaginatedResponse[EventSummary]
