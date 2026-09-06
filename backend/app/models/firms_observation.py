"""NASA FIRMS thermal observation rows (Phase 2 NRT persistence).

Semantics: these are satellite thermal hotspot observations awaiting
incremental event formation. They are NOT fires, industrial fires, alerts,
or confirmed sources. ``event_id`` stays NULL until a later phase assigns
events.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from geoalchemy2 import Geometry
from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class FirmsObservation(Base):
    """
    One normalized FIRMS hotspot observation.

    Identity is ``observation_hash`` (local SHA-256 over stable FIRMS fields),
    not a NASA-native observation ID.
    """

    __tablename__ = "firms_observations"
    __table_args__ = (
        UniqueConstraint("observation_hash", name="uq_firms_observations_observation_hash"),
        Index("ix_firms_observations_acq_datetime", "acq_datetime"),
        Index("ix_firms_observations_event_id", "event_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    observation_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    geometry = mapped_column(
        Geometry(geometry_type="POINT", srid=4326, spatial_index=False),
        nullable=False,
    )

    acq_date: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    acq_time: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    acq_datetime: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    satellite: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    instrument: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    confidence: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    bright_ti4: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bright_ti5: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    scan: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    track: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    frp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    daynight: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    source_file: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)

    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Remains NULL until a later incremental event-formation phase.
    event_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
