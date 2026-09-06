"""Thermal events from frozen Stage VI risk-prioritization output."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from geoalchemy2 import Geometry
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ThermalEvent(Base):
    """
    One ST-DBSCAN thermal event enriched through Stages G → VI.

    Semantics preserved from AIML:
    - risk_score is an engineering decision-support score, not a fire probability.
    - investigation_priority is investigation priority, not confirmed fire status.
    - Missing STA/environmental evidence remains unavailable (null / false flags),
      never converted into negative evidence.
    - Facility association is spatial attribution, not source classification.
    """

    __tablename__ = "thermal_events"
    __table_args__ = (
        UniqueConstraint("event_id", name="uq_thermal_events_event_id"),
        Index("ix_thermal_events_event_start", "event_start"),
        Index("ix_thermal_events_investigation_priority", "investigation_priority"),
        Index("ix_thermal_events_industrial_context", "industrial_context"),
        Index("ix_thermal_events_facility_id", "facility_id"),
        Index("ix_thermal_events_facility_type", "facility_type"),
        Index("ix_thermal_events_persistence_label", "persistence_label"),
        Index("ix_thermal_events_anomaly_status", "anomaly_status"),
        Index("ix_thermal_events_risk_score", "risk_score"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(64), nullable=False)

    # --- temporal ---
    event_start: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    event_end: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    observed_duration_hours: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    distinct_detection_days: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    span_days: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    duty_cycle: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mean_gap_hours: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_gap_hours: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    detection_frequency_per_day: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # --- thermal ---
    detection_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    peak_frp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mean_frp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    median_frp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_frp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    day_detection_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    night_detection_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # --- location ---
    centroid_latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    centroid_longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    min_latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    min_longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    centroid_wkt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    footprint_wkt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    geometry = mapped_column(
        Geometry(geometry_type="POINT", srid=4326, spatial_index=False),
        nullable=True,
    )
    footprint_geometry = mapped_column(
        Geometry(geometry_type="GEOMETRY", srid=4326, spatial_index=False),
        nullable=True,
    )

    # --- persistence (Stage G.1) ---
    persistence_label: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    persistence_basis: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # --- facility association (Stage I.2) ---
    facility_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    facility_name: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    facility_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    facility_association_method: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    facility_attribution_confidence: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    facility_distance_km: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    candidate_facility_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # --- STA evidence (Stage I.5) — supporting evidence only, not ground truth ---
    sta_association_status: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    primary_sta_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    sta_layer_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    sta_match_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    sta_nearest_distance_km: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sta_intersection_area_m2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sta_evidence_available: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    sta_temporal_relation: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    sta_evidence_quality: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    # --- historical baseline / anomaly (Stages I.3 / I.4) ---
    baseline_observation_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    baseline_history_status: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    anomaly_unavailable_reason: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    anomaly_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    anomaly_status: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    anomaly_confidence: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    peak_frp_deviation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    event_size_deviation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    duration_deviation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    distance_deviation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    persistence_deviation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    monthly_deviation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    features_available: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    features_evaluated: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    anomaly_explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # --- environmental context (Stage I.6); null/false = unavailable evidence ---
    # Context/evidence only — not industrial-fire classification or risk.
    landcover_available: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    landcover_source: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    landcover_year: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    dominant_landcover_class: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    dominant_landcover_fraction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    landcover_class_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    vegetation_context_available: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    vegetation_present: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    vegetation_coverage_fraction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    distance_to_vegetation_km: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    builtup_context_available: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    builtup_present: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    builtup_coverage_fraction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    distance_to_builtup_km: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    water_context_available: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    water_present: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    water_coverage_fraction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    distance_to_water_km: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    agriculture_context_available: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    agriculture_present: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    agriculture_coverage_fraction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    distance_to_agriculture_km: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    satellite_context_available: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    satellite_source: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    satellite_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    satellite_value_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    # --- evidence fusion (Stage I.7) ---
    temporal_evidence_available: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    infrastructure_evidence_available: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    sta_domain_available: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    environmental_domain_available: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    temporal_persistence_signal: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    temporal_anomaly_signal: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    temporal_evidence_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    infrastructure_association_signal: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    infrastructure_facility_type_signal: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    infrastructure_confidence_signal: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    infrastructure_history_signal: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    infrastructure_evidence_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    sta_association_signal: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    sta_layer_signal: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    sta_quality_signal: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    sta_evidence_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    environmental_landcover_signal: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    environmental_vegetation_signal: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    environmental_agriculture_signal: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    environmental_builtup_signal: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    environmental_water_signal: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    environmental_evidence_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    evidence_sources_present_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    evidence_sources_present: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    evidence_sources_missing: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    evidence_availability_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    evidence_conflict_flag: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    evidence_conflict_codes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    evidence_conflict_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    infrastructure_evidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    temporal_evidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    historical_evidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    anomaly_evidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sta_evidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    environmental_evidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    industrial_evidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    environmental_support_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    evidence_fusion_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    evidence_coverage: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    evidence_strength: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    evidence_profile_codes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    supporting_evidence_codes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ambiguous_evidence_codes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    limiting_evidence_codes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    source_intelligence_candidate: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    candidate_rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    candidate_is_ground_truth: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    evidence_sufficiency: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    evidence_uncertainty: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    interpretation_confidence: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # --- risk prioritization (Stage VI) ---
    risk_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    investigation_priority: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    recommended_action: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    industrial_context: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    thermal_severity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    thermal_severity_band: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    persistence_priority_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    persistence_priority_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    anomaly_priority_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    anomaly_priority_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    facility_context_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    facility_context_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    industrial_evidence_component: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    uncertainty_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    uncertainty_band: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    dominant_risk_factors: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    dominant_uncertainty_factors: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    priority_reasons: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    priority_warnings: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    risk_limiting_evidence_codes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    risk_scoring_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # --- Phase 3 realtime lifecycle (does not change Stage VI semantics) ---
    # is_active: still within temporal continuity for accepting NRT observations.
    # This is NOT "confirmed fire" / emergency status.
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    last_detection_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Running count of detections with valid FRP (for correct mean_frp updates).
    frp_valid_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
