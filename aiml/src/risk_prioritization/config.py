"""
Configuration for GIFT Stage VI (Decision & Risk Prioritization).

All weights and thresholds are ENGINEERING DEFAULTS.
They are NOT scientifically validated and must not be presented as
calibrated fire probabilities or validated risk accuracy.

Stage V currently has VALIDATION_DATA_UNAVAILABLE — therefore Stage VI
makes no validated performance claims.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

RISK_SCORING_VERSION = "VI.1.0"

# Priority / severity / uncertainty bands
PRIORITY_LOW = "LOW"
PRIORITY_MEDIUM = "MEDIUM"
PRIORITY_HIGH = "HIGH"
PRIORITY_CRITICAL = "CRITICAL"

SEVERITY_LOW = "LOW"
SEVERITY_MODERATE = "MODERATE"
SEVERITY_HIGH = "HIGH"
SEVERITY_EXTREME = "EXTREME"

UNCERTAINTY_LOW = "LOW"
UNCERTAINTY_MODERATE = "MODERATE"
UNCERTAINTY_HIGH = "HIGH"
UNCERTAINTY_VERY_HIGH = "VERY_HIGH"

INDUSTRIAL_CONTEXT_STRONG = "STRONG"
INDUSTRIAL_CONTEXT_POSSIBLE = "POSSIBLE"
INDUSTRIAL_CONTEXT_AMBIGUOUS = "AMBIGUOUS"
INDUSTRIAL_CONTEXT_INSUFFICIENT = "INSUFFICIENT"

ACTION_MONITOR = "MONITOR"
ACTION_REVIEW = "REVIEW"
ACTION_PRIORITIZE = "PRIORITIZE_INVESTIGATION"
ACTION_URGENT = "URGENT_REVIEW"


@dataclass(frozen=True)
class RiskPrioritizationConfig:
    """Engineering parameters for Stage VI priority scoring."""

    events_path: Path = Path("data/processed/thermal_events_with_evidence_fusion.csv")
    scoring_version: str = RISK_SCORING_VERSION

    # Component weights (sum of positive max contributions ≈ 100 before uncertainty dampening)
    weight_thermal: float = 25.0
    weight_persistence: float = 15.0
    weight_anomaly: float = 25.0
    weight_facility: float = 15.0
    weight_industrial_evidence: float = 20.0
    # Ambiguity-only operational dampening (NOT missing-STA/env penalty)
    ambiguity_dampening_max: float = 8.0

    # Thermal severity engineering cutoffs on peak_frp (MW) after log1p scaling helpers
    # Bands use robust rank percentiles computed on the batch when possible.
    thermal_frp_log_cap: float = 8.0  # log1p(FRP) cap (~2980 MW)
    thermal_detection_cap: float = 50.0
    thermal_duration_hours_cap: float = 168.0  # 7 days

    # Priority thresholds on risk_score 0-100
    priority_medium_min: float = 25.0
    priority_high_min: float = 50.0
    priority_critical_min: float = 75.0

    # Optional floor: extreme thermal alone can reach at least MEDIUM (documented)
    extreme_thermal_minimum_priority: str = PRIORITY_MEDIUM

    # Facility-type context weights 0-1 (engineering, not hazard claims)
    facility_type_weights: dict[str, float] = field(
        default_factory=lambda: {
            "REFINERY": 1.0,
            "LNG_TERMINAL": 1.0,
            "POWER_PLANT": 0.9,
            "MINE": 0.75,
            "INDUSTRIAL_AREA": 0.6,
            "OTHER_INDUSTRIAL": 0.5,
            "UNKNOWN": 0.35,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["events_path"] = str(self.events_path)
        payload["priority_bands"] = {
            "LOW": f"[0, {self.priority_medium_min})",
            "MEDIUM": f"[{self.priority_medium_min}, {self.priority_high_min})",
            "HIGH": f"[{self.priority_high_min}, {self.priority_critical_min})",
            "CRITICAL": f"[{self.priority_critical_min}, 100]",
        }
        payload["semantics"] = {
            "risk_score": "Deterministic decision-support score 0-100, not a probability.",
            "missing_evidence": "Unavailable evidence is neutral; never treated as negative industrial evidence.",
            "validation": (
                "Risk prioritization has not been independently validated because "
                "Stage V currently has no independent reference dataset."
            ),
        }
        return payload
