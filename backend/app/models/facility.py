"""Normalized facility universe from Stage I.1 (`osm_facilities.csv`)."""

from __future__ import annotations

from typing import Optional

from geoalchemy2 import Geometry
from sqlalchemy import Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Facility(Base):
    """
    OSM-derived industrial facility record.

    Source: frozen Stage I.1 normalized facility extract — not reconstructed
    from thermal-event associations alone.
    """

    __tablename__ = "facilities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    facility_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)

    facility_name: Mapped[Optional[str]] = mapped_column(String(512), nullable=True, index=True)
    facility_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    industrial_subtype: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    operator: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    landuse: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    power_type: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    man_made_type: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    confidence: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    geometry_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    geometry = mapped_column(
        Geometry(geometry_type="POINT", srid=4326, spatial_index=False),
        nullable=True,
    )
    geometry_wkt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    osm_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    osm_type: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    osm_tags = mapped_column(JSONB, nullable=True)

    source: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    source_version: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
