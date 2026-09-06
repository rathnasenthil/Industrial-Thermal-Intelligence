"""Align evidence_coverage with batch I.7 string labels (Phase 10).

Revision ID: 007_evidence_coverage_str
Revises: 006_i6_env_context_cols
Create Date: 2026-09-06

Batch Stage I.7 writes evidence_coverage as present/total labels (e.g. \"3/4\"),
not a float fraction. Initial schema incorrectly used Float.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "007_evidence_coverage_str"
down_revision: Union[str, Sequence[str], None] = "006_i6_env_context_cols"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop float values that cannot represent \"3/4\" labels; column was unused
    # for realtime I.7 until Phase 10. Historical Stage VI CSV floats (if any)
    # become null via USING cast only when numeric — otherwise null.
    op.execute(
        """
        ALTER TABLE thermal_events
        ALTER COLUMN evidence_coverage TYPE VARCHAR(32)
        USING (
            CASE
                WHEN evidence_coverage IS NULL THEN NULL
                ELSE trim(both from evidence_coverage::text)
            END
        )
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE thermal_events
        ALTER COLUMN evidence_coverage TYPE DOUBLE PRECISION
        USING (
            CASE
                WHEN evidence_coverage ~ '^[0-9]+(\\.[0-9]+)?$'
                    THEN evidence_coverage::double precision
                WHEN evidence_coverage ~ '^[0-9]+/[0-9]+$'
                    THEN split_part(evidence_coverage, '/', 1)::double precision
                       / NULLIF(split_part(evidence_coverage, '/', 2)::double precision, 0)
                ELSE NULL
            END
        )
        """
    )
