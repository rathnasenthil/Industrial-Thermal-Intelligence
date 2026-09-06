"""Add firms_observations for Phase 2 NRT persistence.

Revision ID: 002_realtime_observations
Revises: 001_initial_schema
Create Date: 2026-09-06

Creates only ``firms_observations``. Does not alter thermal_events,
facilities, or event_facility_candidates. Does not create event_detections
or ingestion_runs.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from geoalchemy2 import Geometry

revision: str = "002_realtime_observations"
down_revision: Union[str, Sequence[str], None] = "001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "firms_observations",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("observation_hash", sa.String(length=64), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column(
            "geometry",
            Geometry(geometry_type="POINT", srid=4326, spatial_index=False),
            nullable=False,
        ),
        sa.Column("acq_date", sa.String(length=32), nullable=True),
        sa.Column("acq_time", sa.String(length=16), nullable=True),
        sa.Column("acq_datetime", sa.DateTime(timezone=True), nullable=True),
        sa.Column("satellite", sa.String(length=32), nullable=True),
        sa.Column("instrument", sa.String(length=32), nullable=True),
        sa.Column("confidence", sa.String(length=32), nullable=True),
        sa.Column("version", sa.String(length=32), nullable=True),
        sa.Column("bright_ti4", sa.Float(), nullable=True),
        sa.Column("bright_ti5", sa.Float(), nullable=True),
        sa.Column("scan", sa.Float(), nullable=True),
        sa.Column("track", sa.Float(), nullable=True),
        sa.Column("frp", sa.Float(), nullable=True),
        sa.Column("daynight", sa.String(length=8), nullable=True),
        sa.Column("type", sa.String(length=32), nullable=True),
        sa.Column("source_file", sa.String(length=256), nullable=True),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("event_id", sa.String(length=64), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "observation_hash", name="uq_firms_observations_observation_hash"
        ),
    )
    op.create_index(
        "ix_firms_observations_acq_datetime",
        "firms_observations",
        ["acq_datetime"],
    )
    op.create_index(
        "ix_firms_observations_event_id",
        "firms_observations",
        ["event_id"],
    )
    op.execute(
        "CREATE INDEX ix_firms_observations_geometry "
        "ON firms_observations USING GIST (geometry)"
    )


def downgrade() -> None:
    op.drop_table("firms_observations")
