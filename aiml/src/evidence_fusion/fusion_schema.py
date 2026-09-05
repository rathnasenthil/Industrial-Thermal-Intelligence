"""
Canonical I.7 evidence-fusion field schema.

Append-only columns. Prior-stage fields are never rewritten.
Missing upstream domains → availability=False + UNAVAILABLE signals
(not fabricated zeros or negative scores).
"""

from __future__ import annotations

from typing import Any

# Columns appended by Stage I.7 (deterministic order).
FUSION_COLUMNS: tuple[str, ...] = (
    # Temporal (G.1 + I.4)
    "temporal_evidence_available",
    "temporal_persistence_signal",
    "temporal_anomaly_signal",
    "temporal_evidence_summary",
    # Infrastructure (I.2 + I.3/I.4 history)
    "infrastructure_evidence_available",
    "infrastructure_association_signal",
    "infrastructure_facility_type_signal",
    "infrastructure_confidence_signal",
    "infrastructure_history_signal",
    "infrastructure_evidence_summary",
    # NASA STA (I.5)
    "sta_domain_available",
    "sta_association_signal",
    "sta_layer_signal",
    "sta_quality_signal",
    "sta_evidence_summary",
    # Environmental (I.6)
    "environmental_domain_available",
    "environmental_landcover_signal",
    "environmental_vegetation_signal",
    "environmental_agriculture_signal",
    "environmental_builtup_signal",
    "environmental_water_signal",
    "environmental_evidence_summary",
    # Availability profile
    "evidence_sources_present_count",
    "evidence_sources_present",
    "evidence_sources_missing",
    "evidence_availability_summary",
    # Conflicts
    "evidence_conflict_flag",
    "evidence_conflict_codes",
    "evidence_conflict_summary",
    # Ordinal multi-family scores (NOT probabilities)
    "infrastructure_evidence_score",
    "temporal_evidence_score",
    "historical_evidence_score",
    "anomaly_evidence_score",
    "sta_evidence_score",
    "environmental_evidence_score",
    "industrial_evidence_score",
    "environmental_support_score",
    "evidence_fusion_score",
    "evidence_coverage",
    "evidence_strength",
    # Structured profile + supporting codes
    "evidence_profile_codes",
    "supporting_evidence_codes",
    "ambiguous_evidence_codes",
    "limiting_evidence_codes",
    # Candidate interpretation (NOT ground truth)
    "source_intelligence_candidate",
    "candidate_rationale",
    "candidate_is_ground_truth",
    "evidence_sufficiency",
    "evidence_uncertainty",
    "interpretation_confidence",
)

I4_IMMUTABLE_COLUMNS: tuple[str, ...] = (
    "anomaly_score",
    "anomaly_status",
    "anomaly_confidence",
    "peak_frp_deviation",
    "event_size_deviation",
    "duration_deviation",
    "distance_deviation",
    "persistence_deviation",
    "monthly_deviation",
)

I5_IMMUTABLE_COLUMNS: tuple[str, ...] = (
    "sta_association_status",
    "primary_sta_id",
    "sta_layer_type",
    "sta_match_count",
    "sta_nearest_distance_km",
    "sta_intersection_area_m2",
    "sta_evidence_available",
    "sta_temporal_relation",
    "sta_evidence_quality",
)

I6_IMMUTABLE_PREFIXES: tuple[str, ...] = (
    "landcover_",
    "vegetation_",
    "builtup_",
    "water_",
    "agriculture_",
    "satellite_",
    "distance_to_",
    "dominant_landcover_",
)

# Forbidden definitive classification / risk field names.
FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "industrial_fire",
    "wildfire",
    "agricultural_fire",
    "source_class",
    "fire_type",
    "risk_score",
    "industrial_probability",
    "pseudo_label",
    "ground_truth_label",
)

# Upstream columns expected when domains are present.
TEMPORAL_INPUT_COLUMNS: tuple[str, ...] = (
    "persistence_label",
    "anomaly_status",
)

INFRASTRUCTURE_INPUT_COLUMNS: tuple[str, ...] = (
    "facility_association_method",
    "facility_attribution_confidence",
    "facility_type",
    "baseline_history_status",
)

STA_INPUT_COLUMNS: tuple[str, ...] = (
    "sta_association_status",
    "sta_evidence_quality",
    "sta_layer_type",
)

ENV_AVAILABILITY_COLUMNS: tuple[str, ...] = (
    "landcover_available",
    "vegetation_context_available",
    "builtup_context_available",
    "water_context_available",
    "agriculture_context_available",
    "satellite_context_available",
)


def _is_nullish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # NaN
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def clean_text(value: Any, default: str | None = None) -> str | None:
    """Normalize a scalar to a clean string or default; never 'nan'."""
    if _is_nullish(value):
        return default
    return str(value).strip()
