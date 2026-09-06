"""Link table: one FIRMS observation belongs to at most one thermal event."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class EventDetection(Base):
    """
    Observation ↔ event membership.

    Does not duplicate FIRMS payload columns — join via observation_hash.
    A thermal event is a spatio-temporal cluster, not a confirmed fire.
    """

    __tablename__ = "event_detections"
    __table_args__ = (
        UniqueConstraint("observation_hash", name="uq_event_detections_observation_hash"),
        Index("ix_event_detections_event_id", "event_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # Business key consistent with firms_observations.event_id / Stage VII FKs.
    event_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("thermal_events.event_id", ondelete="CASCADE"),
        nullable=False,
    )
    observation_hash: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("firms_observations.observation_hash", ondelete="CASCADE"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
