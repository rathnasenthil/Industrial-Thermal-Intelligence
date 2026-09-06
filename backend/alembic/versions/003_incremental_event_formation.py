"""Incremental event formation schema (Phase 3).

Revision ID: 003_incremental_event_formation
Revises: 002_realtime_observations
Create Date: 2026-09-06

- Adds realtime lifecycle columns on thermal_events
- Initializes historical events as inactive (last_detection_at = event_end)
- Creates event_detections
- Creates sequence for stable EVT_####### allocation
Does NOT truncate or regenerate Stage VI data.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "003_incremental_event_formation"
down_revision: Union[str, Sequence[str], None] = "002_realtime_observations"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "thermal_events",
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.add_column(
        "thermal_events",
        sa.Column("last_detection_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "thermal_events",
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "thermal_events",
        sa.Column("frp_valid_count", sa.Integer(), nullable=True),
    )
    op.create_index("ix_thermal_events_is_active", "thermal_events", ["is_active"])
    op.create_index(
        "ix_thermal_events_last_detection_at",
        "thermal_events",
        ["last_detection_at"],
    )

    # Historical Stage VI events are closed; seed last_detection_at from event_end.
    op.execute(
        """
        UPDATE thermal_events
        SET last_detection_at = event_end,
            is_active = false,
            updated_at = NOW(),
            frp_valid_count = CASE
                WHEN total_frp IS NOT NULL AND mean_frp IS NOT NULL AND mean_frp <> 0
                    THEN GREATEST(1, ROUND(total_frp / mean_frp)::integer)
                WHEN peak_frp IS NOT NULL THEN COALESCE(detection_count, 0)
                ELSE 0
            END
        """
    )

    op.create_table(
        "event_detections",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("event_id", sa.String(length=64), nullable=False),
        sa.Column("observation_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["event_id"], ["thermal_events.event_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["observation_hash"],
            ["firms_observations.observation_hash"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "observation_hash", name="uq_event_detections_observation_hash"
        ),
    )
    op.create_index("ix_event_detections_event_id", "event_detections", ["event_id"])

    # Stable EVT_####### allocator — never renumber existing IDs.
    op.execute("CREATE SEQUENCE IF NOT EXISTS thermal_event_external_id_seq")
    op.execute(
        """
        SELECT setval(
            'thermal_event_external_id_seq',
            GREATEST(
                1,
                COALESCE(
                    (
                        SELECT MAX(
                            CAST(SUBSTRING(event_id FROM 5) AS BIGINT)
                        )
                        FROM thermal_events
                        WHERE event_id ~ '^EVT_[0-9]+$'
                    ),
                    1
                )
            ),
            true
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP SEQUENCE IF EXISTS thermal_event_external_id_seq")
    op.drop_table("event_detections")
    op.drop_index("ix_thermal_events_last_detection_at", table_name="thermal_events")
    op.drop_index("ix_thermal_events_is_active", table_name="thermal_events")
    op.drop_column("thermal_events", "frp_valid_count")
    op.drop_column("thermal_events", "updated_at")
    op.drop_column("thermal_events", "last_detection_at")
    op.drop_column("thermal_events", "is_active")
