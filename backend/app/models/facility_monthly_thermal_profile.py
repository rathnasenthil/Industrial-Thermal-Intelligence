"""Persisted Stage I.3 sparse monthly thermal profile per facility."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class FacilityMonthlyThermalProfile(Base):
    """
    Long-format monthly activity for confirmed I.2 associations only.

    Sparse: no zero-activity months. Not anomaly detection.
    """

    __tablename__ = "facility_monthly_thermal_profile"
    __table_args__ = (
        UniqueConstraint(
            "facility_id",
            "month",
            name="uq_facility_monthly_thermal_profile_facility_month",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    facility_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("facilities.facility_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    month: Mapped[int] = mapped_column(Integer, nullable=False)
    event_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    detection_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    event_fraction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
