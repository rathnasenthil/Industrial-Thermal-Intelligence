"""Facility I.3 fingerprint persistence (Phase 6).

Revision ID: 004_facility_fingerprints
Revises: 003_incremental_event_formation
Create Date: 2026-09-06

Creates:
- facility_thermal_fingerprints (one row per facility)
- facility_monthly_thermal_profile (sparse facility×month)

Does NOT alter facilities identity columns or thermal_events schema.
Does NOT truncate historical Stage VII data.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "004_facility_fingerprints"
down_revision: Union[str, Sequence[str], None] = "003_incremental_event_formation"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "facility_thermal_fingerprints",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("facility_id", sa.String(length=128), nullable=False),
        sa.Column("facility_name", sa.String(length=512), nullable=True),
        sa.Column("facility_type", sa.String(length=64), nullable=True),
        sa.Column("event_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("detection_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("observation_day_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("first_observation_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_observation_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("observation_span_days", sa.Float(), nullable=True),
        sa.Column("active_month_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("day_event_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("night_event_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("day_event_fraction", sa.Float(), nullable=True),
        sa.Column("night_event_fraction", sa.Float(), nullable=True),
        sa.Column("peak_frp_median", sa.Float(), nullable=True),
        sa.Column("peak_frp_mad", sa.Float(), nullable=True),
        sa.Column("peak_frp_p25", sa.Float(), nullable=True),
        sa.Column("peak_frp_p75", sa.Float(), nullable=True),
        sa.Column("peak_frp_p90", sa.Float(), nullable=True),
        sa.Column("peak_frp_max", sa.Float(), nullable=True),
        sa.Column("event_size_median", sa.Float(), nullable=True),
        sa.Column("event_size_mad", sa.Float(), nullable=True),
        sa.Column("event_size_p25", sa.Float(), nullable=True),
        sa.Column("event_size_p75", sa.Float(), nullable=True),
        sa.Column("event_size_p90", sa.Float(), nullable=True),
        sa.Column("event_size_max", sa.Float(), nullable=True),
        sa.Column("duration_hours_median", sa.Float(), nullable=True),
        sa.Column("duration_hours_mad", sa.Float(), nullable=True),
        sa.Column("duration_hours_p25", sa.Float(), nullable=True),
        sa.Column("duration_hours_p75", sa.Float(), nullable=True),
        sa.Column("duration_hours_p90", sa.Float(), nullable=True),
        sa.Column("duration_hours_max", sa.Float(), nullable=True),
        sa.Column("persistent_event_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("persistent_event_fraction", sa.Float(), nullable=True),
        sa.Column("recurring_event_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("recurring_event_fraction", sa.Float(), nullable=True),
        sa.Column("short_lived_event_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("short_lived_event_fraction", sa.Float(), nullable=True),
        sa.Column(
            "insufficient_observations_event_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("distance_km_median", sa.Float(), nullable=True),
        sa.Column("distance_km_mad", sa.Float(), nullable=True),
        sa.Column("distance_km_p25", sa.Float(), nullable=True),
        sa.Column("distance_km_p75", sa.Float(), nullable=True),
        sa.Column("distance_km_p90", sa.Float(), nullable=True),
        sa.Column("distance_km_max", sa.Float(), nullable=True),
        sa.Column("within_facility_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("intersects_facility_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("near_facility_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("high_confidence_event_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("medium_confidence_event_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("low_confidence_event_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "ambiguous_candidate_opportunity_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "fingerprint_observation_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "fingerprint_status",
            sa.String(length=64),
            nullable=False,
            server_default="NO_OBSERVATIONS",
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["facility_id"],
            ["facilities.facility_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("facility_id", name="uq_facility_thermal_fingerprints_facility_id"),
    )
    op.create_index(
        "ix_facility_thermal_fingerprints_facility_id",
        "facility_thermal_fingerprints",
        ["facility_id"],
    )
    op.create_index(
        "ix_facility_thermal_fingerprints_status",
        "facility_thermal_fingerprints",
        ["fingerprint_status"],
    )

    op.create_table(
        "facility_monthly_thermal_profile",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("facility_id", sa.String(length=128), nullable=False),
        sa.Column("month", sa.Integer(), nullable=False),
        sa.Column("event_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("detection_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("event_fraction", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["facility_id"],
            ["facilities.facility_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "facility_id",
            "month",
            name="uq_facility_monthly_thermal_profile_facility_month",
        ),
    )
    op.create_index(
        "ix_facility_monthly_thermal_profile_facility_id",
        "facility_monthly_thermal_profile",
        ["facility_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_facility_monthly_thermal_profile_facility_id",
        table_name="facility_monthly_thermal_profile",
    )
    op.drop_table("facility_monthly_thermal_profile")
    op.drop_index(
        "ix_facility_thermal_fingerprints_status",
        table_name="facility_thermal_fingerprints",
    )
    op.drop_index(
        "ix_facility_thermal_fingerprints_facility_id",
        table_name="facility_thermal_fingerprints",
    )
    op.drop_table("facility_thermal_fingerprints")
