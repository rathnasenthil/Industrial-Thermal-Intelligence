"""Stage I.2 event↔facility candidate associations (including ambiguous NEAR sets)."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import BigInteger, Float, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class EventFacilityCandidate(Base):
    """
    Candidate facility relationships for a thermal event.

    Preserves non-primary candidates so AMBIGUOUS / multi-NEAR cases remain
    inspectable on the event detail page. Relation labels are spatial
    interpretations, not ground-truth industrial classification.
    """

    __tablename__ = "event_facility_candidates"
    __table_args__ = (
        UniqueConstraint(
            "event_id",
            "facility_id",
            name="uq_event_facility_candidates_event_facility",
        ),
        Index("ix_event_facility_candidates_event_id", "event_id"),
        Index("ix_event_facility_candidates_facility_id", "facility_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("thermal_events.event_id", ondelete="CASCADE"),
        nullable=False,
    )
    facility_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("facilities.facility_id", ondelete="CASCADE"),
        nullable=False,
    )
    facility_name: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    facility_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    spatial_relation: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    distance_km: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    candidate_rank: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    candidate_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
