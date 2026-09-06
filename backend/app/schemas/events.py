"""Pydantic schemas for thermal event API responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import GeometryPoint, PaginatedResponse


class EventSummary(BaseModel):
    """List-view event summary. risk_score is decision-support, not probability."""

    model_config = ConfigDict(from_attributes=True)

    event_id: str
    event_start: Optional[datetime] = None
    event_end: Optional[datetime] = None
    observed_duration_hours: Optional[float] = None
    detection_count: Optional[int] = None
    peak_frp: Optional[float] = None
    mean_frp: Optional[float] = None
    latitude: Optional[float] = Field(
        default=None,
        description="Centroid latitude (source: centroid_latitude)",
    )
    longitude: Optional[float] = Field(
        default=None,
        description="Centroid longitude (source: centroid_longitude)",
    )
    geometry: Optional[GeometryPoint] = None
    persistence_label: Optional[str] = Field(
        default=None,
        description="Persistence class from Stage G.1 (API alias: persistence_class)",
    )
    facility_id: Optional[str] = None
    facility_name: Optional[str] = None
    facility_type: Optional[str] = None
    facility_association_method: Optional[str] = None
    facility_distance_km: Optional[float] = None
    anomaly_status: Optional[str] = None
    industrial_context: Optional[str] = None
    risk_score: Optional[float] = Field(
        default=None,
        description="Engineering investigation prioritization score (0ΓÇô100), not a fire probability",
    )
    investigation_priority: Optional[str] = Field(
        default=None,
        description="Investigation priority (API alias: priority). CRITICAL Γëá confirmed fire.",
    )
    thermal_severity_band: Optional[str] = None
    recommended_action: Optional[str] = None


class FacilityCandidateSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    facility_id: str
    facility_name: Optional[str] = None
    facility_type: Optional[str] = None
    spatial_relation: Optional[str] = None
    distance_km: Optional[float] = None
    candidate_rank: Optional[int] = None
    candidate_score: Optional[float] = None


class EventDetail(EventSummary):
    """Full event detail for GET /api/events/{event_id}."""

    distinct_detection_days: Optional[int] = None
    span_days: Optional[float] = None
    duty_cycle: Optional[float] = None
    mean_gap_hours: Optional[float] = None
    max_gap_hours: Optional[float] = None
    median_frp: Optional[float] = None
    total_frp: Optional[float] = None
    day_detection_count: Optional[int] = None
    night_detection_count: Optional[int] = None
    min_latitude: Optional[float] = None
    max_latitude: Optional[float] = None
    min_longitude: Optional[float] = None
    max_longitude: Optional[float] = None
    centroid_wkt: Optional[str] = None
    footprint_wkt: Optional[str] = None
    persistence_basis: Optional[str] = None
    facility_attribution_confidence: Optional[str] = None
    candidate_facility_count: Optional[int] = None
    facility_candidates: list[FacilityCandidateSummary] = Field(default_factory=list)

    baseline_observation_count: Optional[int] = None
    baseline_history_status: Optional[str] = None
    anomaly_unavailable_reason: Optional[str] = None
    anomaly_score: Optional[float] = None
    anomaly_confidence: Optional[str] = None
    anomaly_explanation: Optional[str] = None

    evidence_sufficiency: Optional[str] = None
    evidence_uncertainty: Optional[str] = None
    evidence_strength: Optional[str] = None
    industrial_evidence_score: Optional[float] = None
    evidence_fusion_score: Optional[float] = None
    source_intelligence_candidate: Optional[str] = None
    candidate_rationale: Optional[str] = None
    candidate_is_ground_truth: Optional[bool] = None
    interpretation_confidence: Optional[str] = None

    thermal_severity_score: Optional[float] = None
    uncertainty_score: Optional[float] = None
    uncertainty_band: Optional[str] = None
    dominant_risk_factors: Optional[str] = None
    dominant_uncertainty_factors: Optional[str] = None
    priority_reasons: Optional[str] = None
    priority_warnings: Optional[str] = None
    risk_limiting_evidence_codes: Optional[str] = None
    risk_scoring_version: Optional[str] = None

    semantics_note: str = (
        "risk_score is an engineering decision-support prioritization score, "
        "not a probability of industrial fire. Missing STA/environmental "
        "evidence is unavailable, not negative evidence."
    )


class EvidenceFamilyBlock(BaseModel):
    available: bool
    status: str = Field(
        description="available | unavailable ΓÇö never invent missing STA/env evidence",
    )
    score: Optional[float] = None
    summary: Optional[str] = None
    details: dict[str, Any] = Field(default_factory=dict)


class EventEvidence(BaseModel):
    event_id: str
    temporal: EvidenceFamilyBlock
    infrastructure: EvidenceFamilyBlock
    historical: EvidenceFamilyBlock
    anomaly: EvidenceFamilyBlock
    sta: EvidenceFamilyBlock
    environmental: EvidenceFamilyBlock
    fusion: dict[str, Any]


class EventTimeline(BaseModel):
    event_id: str
    event_start: Optional[datetime] = None
    event_end: Optional[datetime] = None
    observed_duration_hours: Optional[float] = None
    distinct_detection_days: Optional[int] = None
    span_days: Optional[float] = None
    duty_cycle: Optional[float] = None
    mean_gap_hours: Optional[float] = None
    max_gap_hours: Optional[float] = None
    detection_count: Optional[int] = None
    day_detection_count: Optional[int] = None
    night_detection_count: Optional[int] = None
    detection_level_timeline_available: bool = False
    detection_level_timeline_note: str = (
        "Per-detection FIRMS timeline is not stored in the Stage VI backend "
        "dataset. Only event-level temporal aggregates are available."
    )


PaginatedEvents = PaginatedResponse[EventSummary]
PaginatedAlerts = PaginatedResponse[EventSummary]
