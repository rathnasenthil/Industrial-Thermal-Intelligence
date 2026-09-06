"""Stage I.5 STA evidence columns on thermal_events (Phase 8).

Revision ID: 005_sta_evidence_columns
Revises: 004_facility_fingerprints
Create Date: 2026-09-06

Adds batch I.5 append columns (distinct from Stage VI fusion sta_*_signal fields).
Does NOT truncate historical data. Does NOT invent STA values.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "005_sta_evidence_columns"
down_revision: Union[str, Sequence[str], None] = "004_facility_fingerprints"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "thermal_events",
        sa.Column("sta_association_status", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "thermal_events",
        sa.Column("primary_sta_id", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "thermal_events",
        sa.Column("sta_layer_type", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "thermal_events",
        sa.Column("sta_match_count", sa.Integer(), nullable=True),
    )
    op.add_column(
        "thermal_events",
        sa.Column("sta_nearest_distance_km", sa.Float(), nullable=True),
    )
    op.add_column(
        "thermal_events",
        sa.Column("sta_intersection_area_m2", sa.Float(), nullable=True),
    )
    op.add_column(
        "thermal_events",
        sa.Column("sta_evidence_available", sa.Boolean(), nullable=True),
    )
    op.add_column(
        "thermal_events",
        sa.Column("sta_temporal_relation", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "thermal_events",
        sa.Column("sta_evidence_quality", sa.String(length=32), nullable=True),
    )
    op.create_index(
        "ix_thermal_events_sta_association_status",
        "thermal_events",
        ["sta_association_status"],
    )
    op.create_index(
        "ix_thermal_events_sta_evidence_available",
        "thermal_events",
        ["sta_evidence_available"],
    )


def downgrade() -> None:
    op.drop_index("ix_thermal_events_sta_evidence_available", table_name="thermal_events")
    op.drop_index("ix_thermal_events_sta_association_status", table_name="thermal_events")
    op.drop_column("thermal_events", "sta_evidence_quality")
    op.drop_column("thermal_events", "sta_temporal_relation")
    op.drop_column("thermal_events", "sta_evidence_available")
    op.drop_column("thermal_events", "sta_intersection_area_m2")
    op.drop_column("thermal_events", "sta_nearest_distance_km")
    op.drop_column("thermal_events", "sta_match_count")
    op.drop_column("thermal_events", "sta_layer_type")
    op.drop_column("thermal_events", "primary_sta_id")
    op.drop_column("thermal_events", "sta_association_status")
