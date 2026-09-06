"""Persisted Stage I.3 facility thermal fingerprint (descriptive baseline)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class FacilityThermalFingerprint(Base):
    """
    One descriptive I.3 fingerprint per facility.

    Not anomaly detection and not industrial-fire classification.
    Stored separately from Stage I.1 ``facilities`` identity rows.
    """

    __tablename__ = "facility_thermal_fingerprints"
    __table_args__ = (
        UniqueConstraint("facility_id", name="uq_facility_thermal_fingerprints_facility_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    facility_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("facilities.facility_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    facility_name: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    facility_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    event_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    detection_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    observation_day_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    first_observation_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_observation_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    observation_span_days: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    active_month_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    day_event_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    night_event_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    day_event_fraction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    night_event_fraction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    peak_frp_median: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    peak_frp_mad: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    peak_frp_p25: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    peak_frp_p75: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    peak_frp_p90: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    peak_frp_max: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    event_size_median: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    event_size_mad: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    event_size_p25: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    event_size_p75: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    event_size_p90: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    event_size_max: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    duration_hours_median: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    duration_hours_mad: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    duration_hours_p25: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    duration_hours_p75: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    duration_hours_p90: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    duration_hours_max: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    persistent_event_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    persistent_event_fraction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recurring_event_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    recurring_event_fraction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    short_lived_event_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    short_lived_event_fraction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    insufficient_observations_event_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )

    distance_km_median: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    distance_km_mad: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    distance_km_p25: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    distance_km_p75: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    distance_km_p90: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    distance_km_max: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    within_facility_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    intersects_facility_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    near_facility_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    high_confidence_event_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    medium_confidence_event_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    low_confidence_event_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    ambiguous_candidate_opportunity_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    fingerprint_observation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fingerprint_status: Mapped[str] = mapped_column(String(64), nullable=False, default="NO_OBSERVATIONS")

    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
