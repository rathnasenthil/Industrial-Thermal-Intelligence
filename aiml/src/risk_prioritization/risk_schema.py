"""Canonical Stage VI output field schema (append-only)."""

from __future__ import annotations

RISK_APPEND_COLUMNS: tuple[str, ...] = (
    "risk_score",
    "investigation_priority",
    "recommended_action",
    "industrial_context",
    "thermal_severity_score",
    "thermal_severity_band",
    "persistence_priority_score",
    "persistence_priority_reason",
    "anomaly_priority_score",
    "anomaly_priority_reason",
    "facility_context_score",
    "facility_context_reason",
    "industrial_evidence_component",
    "uncertainty_score",
    "uncertainty_band",
    "dominant_risk_factors",
    "dominant_uncertainty_factors",
    "priority_reasons",
    "priority_warnings",
    "risk_limiting_evidence_codes",
    "risk_scoring_version",
)

I7_IMMUTABLE_COLUMNS: tuple[str, ...] = (
    "industrial_evidence_score",
    "evidence_strength",
    "evidence_sufficiency",
    "source_intelligence_candidate",
    "evidence_uncertainty",
    "candidate_is_ground_truth",
    "anomaly_score",
    "anomaly_status",
    "persistence_label",
    "facility_association_method",
)
