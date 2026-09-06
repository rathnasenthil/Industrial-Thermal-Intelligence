"""Initial PostGIS schema for facilities, thermal events, and candidates.

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-09-05
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from geoalchemy2 import Geometry
from sqlalchemy.dialects import postgresql

revision: str = "001_initial_schema"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")

    op.create_table(
        "facilities",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("facility_id", sa.String(length=128), nullable=False),
        sa.Column("facility_name", sa.String(length=512), nullable=True),
        sa.Column("facility_type", sa.String(length=64), nullable=True),
        sa.Column("industrial_subtype", sa.String(length=128), nullable=True),
        sa.Column("operator", sa.String(length=256), nullable=True),
        sa.Column("landuse", sa.String(length=128), nullable=True),
        sa.Column("power_type", sa.String(length=128), nullable=True),
        sa.Column("man_made_type", sa.String(length=128), nullable=True),
        sa.Column("confidence", sa.String(length=32), nullable=True),
        sa.Column("geometry_type", sa.String(length=32), nullable=True),
        sa.Column("latitude", sa.Float(), nullable=True),
        sa.Column("longitude", sa.Float(), nullable=True),
        sa.Column(
            "geometry",
            Geometry(geometry_type="POINT", srid=4326, spatial_index=False),
            nullable=True,
        ),
        sa.Column("geometry_wkt", sa.Text(), nullable=True),
        sa.Column("osm_id", sa.String(length=64), nullable=True),
        sa.Column("osm_type", sa.String(length=16), nullable=True),
        sa.Column("osm_tags", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("source", sa.String(length=128), nullable=True),
        sa.Column("source_version", sa.String(length=256), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("facility_id"),
    )
    op.create_index("ix_facilities_facility_id", "facilities", ["facility_id"])
    op.create_index("ix_facilities_facility_name", "facilities", ["facility_name"])
    op.create_index("ix_facilities_facility_type", "facilities", ["facility_type"])
    op.create_index("ix_facilities_osm_id", "facilities", ["osm_id"])
    op.execute(
        "CREATE INDEX ix_facilities_geometry ON facilities USING GIST (geometry)"
    )

    op.create_table(
        "thermal_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("event_id", sa.String(length=64), nullable=False),
        sa.Column("event_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("event_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("observed_duration_hours", sa.Float(), nullable=True),
        sa.Column("distinct_detection_days", sa.Integer(), nullable=True),
        sa.Column("span_days", sa.Float(), nullable=True),
        sa.Column("duty_cycle", sa.Float(), nullable=True),
        sa.Column("mean_gap_hours", sa.Float(), nullable=True),
        sa.Column("max_gap_hours", sa.Float(), nullable=True),
        sa.Column("detection_frequency_per_day", sa.Float(), nullable=True),
        sa.Column("detection_count", sa.Integer(), nullable=True),
        sa.Column("peak_frp", sa.Float(), nullable=True),
        sa.Column("mean_frp", sa.Float(), nullable=True),
        sa.Column("median_frp", sa.Float(), nullable=True),
        sa.Column("total_frp", sa.Float(), nullable=True),
        sa.Column("day_detection_count", sa.Integer(), nullable=True),
        sa.Column("night_detection_count", sa.Integer(), nullable=True),
        sa.Column("centroid_latitude", sa.Float(), nullable=True),
        sa.Column("centroid_longitude", sa.Float(), nullable=True),
        sa.Column("min_latitude", sa.Float(), nullable=True),
        sa.Column("max_latitude", sa.Float(), nullable=True),
        sa.Column("min_longitude", sa.Float(), nullable=True),
        sa.Column("max_longitude", sa.Float(), nullable=True),
        sa.Column("centroid_wkt", sa.Text(), nullable=True),
        sa.Column("footprint_wkt", sa.Text(), nullable=True),
        sa.Column(
            "geometry",
            Geometry(geometry_type="POINT", srid=4326, spatial_index=False),
            nullable=True,
        ),
        sa.Column(
            "footprint_geometry",
            Geometry(geometry_type="GEOMETRY", srid=4326, spatial_index=False),
            nullable=True,
        ),
        sa.Column("persistence_label", sa.String(length=64), nullable=True),
        sa.Column("persistence_basis", sa.Text(), nullable=True),
        sa.Column("facility_id", sa.String(length=128), nullable=True),
        sa.Column("facility_name", sa.String(length=512), nullable=True),
        sa.Column("facility_type", sa.String(length=64), nullable=True),
        sa.Column("facility_association_method", sa.String(length=64), nullable=True),
        sa.Column("facility_attribution_confidence", sa.String(length=32), nullable=True),
        sa.Column("facility_distance_km", sa.Float(), nullable=True),
        sa.Column("candidate_facility_count", sa.Integer(), nullable=True),
        sa.Column("baseline_observation_count", sa.Integer(), nullable=True),
        sa.Column("baseline_history_status", sa.String(length=64), nullable=True),
        sa.Column("anomaly_unavailable_reason", sa.String(length=128), nullable=True),
        sa.Column("anomaly_score", sa.Float(), nullable=True),
        sa.Column("anomaly_status", sa.String(length=64), nullable=True),
        sa.Column("anomaly_confidence", sa.String(length=32), nullable=True),
        sa.Column("peak_frp_deviation", sa.Float(), nullable=True),
        sa.Column("event_size_deviation", sa.Float(), nullable=True),
        sa.Column("duration_deviation", sa.Float(), nullable=True),
        sa.Column("distance_deviation", sa.Float(), nullable=True),
        sa.Column("persistence_deviation", sa.Float(), nullable=True),
        sa.Column("monthly_deviation", sa.Float(), nullable=True),
        sa.Column("features_available", sa.Integer(), nullable=True),
        sa.Column("features_evaluated", sa.Integer(), nullable=True),
        sa.Column("anomaly_explanation", sa.Text(), nullable=True),
        sa.Column("landcover_available", sa.Boolean(), nullable=True),
        sa.Column("vegetation_context_available", sa.Boolean(), nullable=True),
        sa.Column("builtup_context_available", sa.Boolean(), nullable=True),
        sa.Column("water_context_available", sa.Boolean(), nullable=True),
        sa.Column("agriculture_context_available", sa.Boolean(), nullable=True),
        sa.Column("satellite_context_available", sa.Boolean(), nullable=True),
        sa.Column("temporal_evidence_available", sa.Boolean(), nullable=True),
        sa.Column("infrastructure_evidence_available", sa.Boolean(), nullable=True),
        sa.Column("sta_domain_available", sa.Boolean(), nullable=True),
        sa.Column("environmental_domain_available", sa.Boolean(), nullable=True),
        sa.Column("temporal_persistence_signal", sa.String(length=64), nullable=True),
        sa.Column("temporal_anomaly_signal", sa.String(length=64), nullable=True),
        sa.Column("temporal_evidence_summary", sa.Text(), nullable=True),
        sa.Column("infrastructure_association_signal", sa.String(length=64), nullable=True),
        sa.Column("infrastructure_facility_type_signal", sa.String(length=64), nullable=True),
        sa.Column("infrastructure_confidence_signal", sa.String(length=64), nullable=True),
        sa.Column("infrastructure_history_signal", sa.String(length=64), nullable=True),
        sa.Column("infrastructure_evidence_summary", sa.Text(), nullable=True),
        sa.Column("sta_association_signal", sa.String(length=64), nullable=True),
        sa.Column("sta_layer_signal", sa.String(length=64), nullable=True),
        sa.Column("sta_quality_signal", sa.String(length=64), nullable=True),
        sa.Column("sta_evidence_summary", sa.Text(), nullable=True),
        sa.Column("environmental_landcover_signal", sa.String(length=64), nullable=True),
        sa.Column("environmental_vegetation_signal", sa.String(length=64), nullable=True),
        sa.Column("environmental_agriculture_signal", sa.String(length=64), nullable=True),
        sa.Column("environmental_builtup_signal", sa.String(length=64), nullable=True),
        sa.Column("environmental_water_signal", sa.String(length=64), nullable=True),
        sa.Column("environmental_evidence_summary", sa.Text(), nullable=True),
        sa.Column("evidence_sources_present_count", sa.Integer(), nullable=True),
        sa.Column("evidence_sources_present", sa.Text(), nullable=True),
        sa.Column("evidence_sources_missing", sa.Text(), nullable=True),
        sa.Column("evidence_availability_summary", sa.Text(), nullable=True),
        sa.Column("evidence_conflict_flag", sa.Boolean(), nullable=True),
        sa.Column("evidence_conflict_codes", sa.Text(), nullable=True),
        sa.Column("evidence_conflict_summary", sa.Text(), nullable=True),
        sa.Column("infrastructure_evidence_score", sa.Float(), nullable=True),
        sa.Column("temporal_evidence_score", sa.Float(), nullable=True),
        sa.Column("historical_evidence_score", sa.Float(), nullable=True),
        sa.Column("anomaly_evidence_score", sa.Float(), nullable=True),
        sa.Column("sta_evidence_score", sa.Float(), nullable=True),
        sa.Column("environmental_evidence_score", sa.Float(), nullable=True),
        sa.Column("industrial_evidence_score", sa.Float(), nullable=True),
        sa.Column("environmental_support_score", sa.Float(), nullable=True),
        sa.Column("evidence_fusion_score", sa.Float(), nullable=True),
        sa.Column("evidence_coverage", sa.Float(), nullable=True),
        sa.Column("evidence_strength", sa.String(length=64), nullable=True),
        sa.Column("evidence_profile_codes", sa.Text(), nullable=True),
        sa.Column("supporting_evidence_codes", sa.Text(), nullable=True),
        sa.Column("ambiguous_evidence_codes", sa.Text(), nullable=True),
        sa.Column("limiting_evidence_codes", sa.Text(), nullable=True),
        sa.Column("source_intelligence_candidate", sa.String(length=64), nullable=True),
        sa.Column("candidate_rationale", sa.Text(), nullable=True),
        sa.Column("candidate_is_ground_truth", sa.Boolean(), nullable=True),
        sa.Column("evidence_sufficiency", sa.String(length=64), nullable=True),
        sa.Column("evidence_uncertainty", sa.String(length=64), nullable=True),
        sa.Column("interpretation_confidence", sa.String(length=64), nullable=True),
        sa.Column("risk_score", sa.Float(), nullable=True),
        sa.Column("investigation_priority", sa.String(length=32), nullable=True),
        sa.Column("recommended_action", sa.String(length=128), nullable=True),
        sa.Column("industrial_context", sa.String(length=64), nullable=True),
        sa.Column("thermal_severity_score", sa.Float(), nullable=True),
        sa.Column("thermal_severity_band", sa.String(length=32), nullable=True),
        sa.Column("persistence_priority_score", sa.Float(), nullable=True),
        sa.Column("persistence_priority_reason", sa.Text(), nullable=True),
        sa.Column("anomaly_priority_score", sa.Float(), nullable=True),
        sa.Column("anomaly_priority_reason", sa.Text(), nullable=True),
        sa.Column("facility_context_score", sa.Float(), nullable=True),
        sa.Column("facility_context_reason", sa.Text(), nullable=True),
        sa.Column("industrial_evidence_component", sa.Float(), nullable=True),
        sa.Column("uncertainty_score", sa.Float(), nullable=True),
        sa.Column("uncertainty_band", sa.String(length=32), nullable=True),
        sa.Column("dominant_risk_factors", sa.Text(), nullable=True),
        sa.Column("dominant_uncertainty_factors", sa.Text(), nullable=True),
        sa.Column("priority_reasons", sa.Text(), nullable=True),
        sa.Column("priority_warnings", sa.Text(), nullable=True),
        sa.Column("risk_limiting_evidence_codes", sa.Text(), nullable=True),
        sa.Column("risk_scoring_version", sa.String(length=64), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("event_id", name="uq_thermal_events_event_id"),
    )
    op.create_index("ix_thermal_events_event_start", "thermal_events", ["event_start"])
    op.create_index(
        "ix_thermal_events_investigation_priority",
        "thermal_events",
        ["investigation_priority"],
    )
    op.create_index(
        "ix_thermal_events_industrial_context",
        "thermal_events",
        ["industrial_context"],
    )
    op.create_index("ix_thermal_events_facility_id", "thermal_events", ["facility_id"])
    op.create_index("ix_thermal_events_facility_type", "thermal_events", ["facility_type"])
    op.create_index(
        "ix_thermal_events_persistence_label",
        "thermal_events",
        ["persistence_label"],
    )
    op.create_index("ix_thermal_events_anomaly_status", "thermal_events", ["anomaly_status"])
    op.create_index("ix_thermal_events_risk_score", "thermal_events", ["risk_score"])
    op.execute(
        "CREATE INDEX ix_thermal_events_geometry ON thermal_events USING GIST (geometry)"
    )
    op.execute(
        "CREATE INDEX ix_thermal_events_footprint_geometry "
        "ON thermal_events USING GIST (footprint_geometry)"
    )

    op.create_table(
        "event_facility_candidates",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("event_id", sa.String(length=64), nullable=False),
        sa.Column("facility_id", sa.String(length=128), nullable=False),
        sa.Column("facility_name", sa.String(length=512), nullable=True),
        sa.Column("facility_type", sa.String(length=64), nullable=True),
        sa.Column("spatial_relation", sa.String(length=64), nullable=True),
        sa.Column("distance_km", sa.Float(), nullable=True),
        sa.Column("candidate_rank", sa.Integer(), nullable=True),
        sa.Column("candidate_score", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["event_id"], ["thermal_events.event_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["facility_id"], ["facilities.facility_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "event_id",
            "facility_id",
            name="uq_event_facility_candidates_event_facility",
        ),
    )
    op.create_index(
        "ix_event_facility_candidates_event_id",
        "event_facility_candidates",
        ["event_id"],
    )
    op.create_index(
        "ix_event_facility_candidates_facility_id",
        "event_facility_candidates",
        ["facility_id"],
    )


def downgrade() -> None:
    op.drop_table("event_facility_candidates")
    op.drop_table("thermal_events")
    op.drop_table("facilities")
