"""Stage I.6 environmental context detail columns (Phase 9).

Revision ID: 006_i6_env_context_cols
Revises: 005_sta_evidence_columns
Create Date: 2026-09-06

Adds batch I.6 ALL_CONTEXT_COLUMNS detail fields that were not present in the
initial schema (availability flags already existed from Stage VI ingest).

Does NOT truncate historical data. Does NOT invent environmental values.
Does NOT add Stage I.7 environmental_*_signal fusion columns (already present).
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "006_i6_env_context_cols"
down_revision: Union[str, Sequence[str], None] = "005_sta_evidence_columns"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "thermal_events",
        sa.Column("landcover_source", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "thermal_events",
        sa.Column("landcover_year", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "thermal_events",
        sa.Column("dominant_landcover_class", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "thermal_events",
        sa.Column("dominant_landcover_fraction", sa.Float(), nullable=True),
    )
    op.add_column(
        "thermal_events",
        sa.Column("landcover_class_count", sa.Integer(), nullable=True),
    )
    op.add_column(
        "thermal_events",
        sa.Column("vegetation_present", sa.Boolean(), nullable=True),
    )
    op.add_column(
        "thermal_events",
        sa.Column("vegetation_coverage_fraction", sa.Float(), nullable=True),
    )
    op.add_column(
        "thermal_events",
        sa.Column("distance_to_vegetation_km", sa.Float(), nullable=True),
    )
    op.add_column(
        "thermal_events",
        sa.Column("builtup_present", sa.Boolean(), nullable=True),
    )
    op.add_column(
        "thermal_events",
        sa.Column("builtup_coverage_fraction", sa.Float(), nullable=True),
    )
    op.add_column(
        "thermal_events",
        sa.Column("distance_to_builtup_km", sa.Float(), nullable=True),
    )
    op.add_column(
        "thermal_events",
        sa.Column("water_present", sa.Boolean(), nullable=True),
    )
    op.add_column(
        "thermal_events",
        sa.Column("water_coverage_fraction", sa.Float(), nullable=True),
    )
    op.add_column(
        "thermal_events",
        sa.Column("distance_to_water_km", sa.Float(), nullable=True),
    )
    op.add_column(
        "thermal_events",
        sa.Column("agriculture_present", sa.Boolean(), nullable=True),
    )
    op.add_column(
        "thermal_events",
        sa.Column("agriculture_coverage_fraction", sa.Float(), nullable=True),
    )
    op.add_column(
        "thermal_events",
        sa.Column("distance_to_agriculture_km", sa.Float(), nullable=True),
    )
    op.add_column(
        "thermal_events",
        sa.Column("satellite_source", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "thermal_events",
        sa.Column("satellite_value", sa.Float(), nullable=True),
    )
    op.add_column(
        "thermal_events",
        sa.Column("satellite_value_name", sa.String(length=128), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("thermal_events", "satellite_value_name")
    op.drop_column("thermal_events", "satellite_value")
    op.drop_column("thermal_events", "satellite_source")
    op.drop_column("thermal_events", "distance_to_agriculture_km")
    op.drop_column("thermal_events", "agriculture_coverage_fraction")
    op.drop_column("thermal_events", "agriculture_present")
    op.drop_column("thermal_events", "distance_to_water_km")
    op.drop_column("thermal_events", "water_coverage_fraction")
    op.drop_column("thermal_events", "water_present")
    op.drop_column("thermal_events", "distance_to_builtup_km")
    op.drop_column("thermal_events", "builtup_coverage_fraction")
    op.drop_column("thermal_events", "builtup_present")
    op.drop_column("thermal_events", "distance_to_vegetation_km")
    op.drop_column("thermal_events", "vegetation_coverage_fraction")
    op.drop_column("thermal_events", "vegetation_present")
    op.drop_column("thermal_events", "landcover_class_count")
    op.drop_column("thermal_events", "dominant_landcover_fraction")
    op.drop_column("thermal_events", "dominant_landcover_class")
    op.drop_column("thermal_events", "landcover_year")
    op.drop_column("thermal_events", "landcover_source")
