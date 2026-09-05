"""
Configuration for GIFT Stage I.7 (Evidence Fusion / Source Intelligence).

All thresholds and ordinal weights are ENGINEERING defaults for transparent,
deterministic fusion. They are NOT scientifically validated and must not be
presented as calibrated fire-source probabilities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# Candidate vocabulary - interpretations, never ground-truth labels.
CANDIDATE_INDUSTRIAL = "INDUSTRIAL_ACTIVITY_CANDIDATE"
CANDIDATE_POSSIBLE_INDUSTRIAL = "POSSIBLE_INDUSTRIAL_ACTIVITY"
CANDIDATE_VEGETATION_CONTEXT = "ENVIRONMENTAL_VEGETATION_CONTEXT"
CANDIDATE_AGRICULTURE_CONTEXT = "ENVIRONMENTAL_AGRICULTURE_CONTEXT"
CANDIDATE_MIXED = "MIXED_OR_CONFLICTING"
CANDIDATE_AMBIGUOUS = "AMBIGUOUS_EVIDENCE"
CANDIDATE_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

CANDIDATE_VALUES: tuple[str, ...] = (
    CANDIDATE_INDUSTRIAL,
    CANDIDATE_POSSIBLE_INDUSTRIAL,
    CANDIDATE_VEGETATION_CONTEXT,
    CANDIDATE_AGRICULTURE_CONTEXT,
    CANDIDATE_MIXED,
    CANDIDATE_AMBIGUOUS,
    CANDIDATE_INSUFFICIENT,
)

# Evidence sufficiency / uncertainty vocabularies.
SUFFICIENCY_SUFFICIENT = "SUFFICIENT_FOR_CANDIDATE"
SUFFICIENCY_PARTIAL = "PARTIAL"
SUFFICIENCY_INSUFFICIENT = "INSUFFICIENT"

UNCERTAINTY_LOW = "LOW"
UNCERTAINTY_MEDIUM = "MEDIUM"
UNCERTAINTY_HIGH = "HIGH"
UNCERTAINTY_VERY_HIGH = "VERY_HIGH"

CONFIRMED_FACILITY_METHODS: frozenset[str] = frozenset(
    {"WITHIN_FACILITY", "INTERSECTS_FACILITY", "NEAR_FACILITY"}
)
STRONG_FACILITY_METHODS: frozenset[str] = frozenset(
    {"WITHIN_FACILITY", "INTERSECTS_FACILITY"}
)
HIGH_FACILITY_CONFIDENCE: frozenset[str] = frozenset({"HIGH", "MEDIUM"})


@dataclass(frozen=True)
class EvidenceFusionConfig:
    """Tunable engineering parameters for Stage I.7 multi-family fusion.

    Attributes:
        events_path: Preferred I.6 events table.
        infra_aggregate_weight: Multiplier for infrastructure family in the
            industrial aggregate (default 2). Engineering default.
        near_min_corroboration_for_possible: Minimum corroboration score
            required before NEAR/weak infrastructure can become
            POSSIBLE_INDUSTRIAL_ACTIVITY (default 2).
        industrial_min_infrastructure: Minimum infrastructure ordinal score
            for INDUSTRIAL_ACTIVITY_CANDIDATE (default 3).
    """

    events_path: Path = Path("data/processed/thermal_events_with_environmental_context.csv")
    infra_aggregate_weight: int = 2
    near_min_corroboration_for_possible: int = 2
    industrial_min_infrastructure: int = 3

    def to_dict(self) -> dict[str, Any]:
        from src.evidence_fusion.evidence_scores import (
            HISTORICAL_EFFECTIVE_CAP,
            INDUSTRIAL_SCORE_MAX,
        )

        payload = asdict(self)
        payload["events_path"] = str(self.events_path)
        payload["industrial_score_max_documented"] = INDUSTRIAL_SCORE_MAX
        payload["historical_effective_cap"] = HISTORICAL_EFFECTIVE_CAP
        payload["ordinal_family_scale"] = {
            "0": "no supporting evidence",
            "1": "weak supporting evidence",
            "2": "moderate supporting evidence",
            "3": "strong supporting evidence",
        }
        return payload

    def describe_rationale(self) -> dict[str, str]:
        from src.evidence_fusion.evidence_scores import INDUSTRIAL_SCORE_MAX

        return {
            "purpose": (
                "I.7 fuses available upstream evidence families via ordinal scores "
                "into an explainable evidence profile and optional source-intelligence "
                "candidate. Candidates are not ground truth."
            ),
            "ordinal_scores": (
                "Family scores use 0-3 ordinal engineering support levels. "
                "They are not probabilities, accuracies, or calibrated confidences."
            ),
            "aggregation": (
                f"industrial_evidence_score = {self.infra_aggregate_weight}*infrastructure "
                "+ temporal_eff + historical_eff(capped) + anomaly_eff(if infra>0) + sta. "
                f"Documented max={INDUSTRIAL_SCORE_MAX}. Missing STA/env add 0, never subtract."
            ),
            "missing_evidence": (
                "Missing or unavailable/null evidence is never treated as a "
                "negative score or as proof of a competing source class."
            ),
            "no_ml": "No machine learning, pseudo-labels, or risk scores.",
            "anomaly_semantics": "ANOMALOUS is not INDUSTRIAL_FIRE.",
            "persistence_semantics": "PERSISTENT is not INDUSTRIAL_FIRE.",
            "facility_semantics": "OSM facility association is not source classification.",
            "sta_semantics": "STA support is not ground truth; STA absence is not anti-industrial.",
            "environmental_semantics": (
                "Environmental context is not a source label; missing env data "
                "is not negative environmental evidence."
            ),
            "candidate_semantics": (
                "I.7 candidate interpretations are deterministic evidence-based "
                "interpretations, not independently validated source labels."
            ),
        }
