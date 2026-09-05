"""
Configuration for GIFT Stage I.4 (Temporal Deviation & Anomaly Detection).

All thresholds and feature weights below are ENGINEERING defaults chosen
for transparency and conservatism. They are NOT scientifically validated
and must not be presented as optimal, calibrated, or clinically certain.
Do not tune them against the same observations being scored (that would
create circular validation).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Anomaly status vocabulary (deterministic).
INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
NORMAL = "NORMAL"
ELEVATED = "ELEVATED"
ANOMALOUS = "ANOMALOUS"

ANOMALY_STATUSES: tuple[str, ...] = (INSUFFICIENT_HISTORY, NORMAL, ELEVATED, ANOMALOUS)

# Confidence vocabulary — evidence quality, NOT probability of fire.
CONFIDENCE_NONE = "NONE"
CONFIDENCE_LOW = "LOW"
CONFIDENCE_MEDIUM = "MEDIUM"
CONFIDENCE_HIGH = "HIGH"

# History quality at scoring time (prior-only, same engineering cutoffs as I.3).
HISTORY_NONE = "NO_PRIOR_OBSERVATIONS"
HISTORY_INSUFFICIENT = "INSUFFICIENT_HISTORY"
HISTORY_LIMITED = "LIMITED_HISTORY"
HISTORY_ESTABLISHED = "ESTABLISHED_BASELINE"

# Reasons facility-specific scoring is unavailable.
REASON_NO_FACILITY = "NO_FACILITY_ASSOCIATION"
REASON_AMBIGUOUS = "AMBIGUOUS_ASSOCIATION"
REASON_INSUFFICIENT_PRIOR = "INSUFFICIENT_PRIOR_HISTORY"

CONFIRMED_ASSOCIATION_METHODS: frozenset[str] = frozenset(
    {"WITHIN_FACILITY", "INTERSECTS_FACILITY", "NEAR_FACILITY"}
)

DEFAULT_FEATURE_WEIGHTS: dict[str, float] = {
    "peak_frp": 0.30,
    "event_size": 0.20,
    "duration": 0.20,
    "distance": 0.10,
    "persistence": 0.10,
    "monthly": 0.10,
}


@dataclass(frozen=True)
class AnomalyConfig:
    """Tunable engineering parameters for Stage I.4.

    Attributes:
        min_observations_for_limited_history: Prior confirmed events needed
            before any conservative deviation scoring is attempted (default 3,
            matching I.3). Below this → anomaly_status=INSUFFICIENT_HISTORY.
        min_observations_for_established_baseline: Prior count at which the
            baseline is treated as ESTABLISHED (default 10, matching I.3).
        min_monthly_prior_observations: Prior same-calendar-month events
            required before a monthly-conditioned deviation is computed.
        normal_max_score: Scores strictly below this are NORMAL.
        elevated_max_score: Scores in [normal_max_score, elevated_max_score)
            are ELEVATED; at/above elevated_max_score are ANOMALOUS.
        zero_mad_constant_mismatch_deviation: When historical MAD is 0 and
            the current value differs from a truly constant historical
            baseline, assign this documented engineering deviation rather
            than dividing by an undocumented epsilon.
        feature_weights: Relative weights for the aggregate score. Only
            features with a non-null deviation contribute; weights of
            contributing features are renormalized to sum to 1.
        events_path / fingerprints_path: Default input paths.
    """

    min_observations_for_limited_history: int = 3
    min_observations_for_established_baseline: int = 10
    min_monthly_prior_observations: int = 3
    normal_max_score: float = 2.0
    elevated_max_score: float = 3.5
    zero_mad_constant_mismatch_deviation: float = 3.0
    feature_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FEATURE_WEIGHTS))
    events_path: Path = Path("data/processed/thermal_events_with_facility_association.csv")
    fingerprints_path: Path = Path("data/processed/facility_thermal_fingerprints.csv")

    def __post_init__(self) -> None:
        if self.min_observations_for_limited_history < 1:
            raise ValueError("min_observations_for_limited_history must be >= 1.")
        if self.min_observations_for_established_baseline <= self.min_observations_for_limited_history:
            raise ValueError(
                "min_observations_for_established_baseline must be strictly greater than "
                "min_observations_for_limited_history."
            )
        if self.normal_max_score <= 0:
            raise ValueError("normal_max_score must be > 0.")
        if self.elevated_max_score <= self.normal_max_score:
            raise ValueError("elevated_max_score must be strictly greater than normal_max_score.")
        if self.zero_mad_constant_mismatch_deviation < 0:
            raise ValueError("zero_mad_constant_mismatch_deviation must be >= 0.")
        if self.min_monthly_prior_observations < 1:
            raise ValueError("min_monthly_prior_observations must be >= 1.")
        weight_sum = sum(self.feature_weights.values())
        if weight_sum <= 0:
            raise ValueError("feature_weights must sum to a positive value.")
        for name, w in self.feature_weights.items():
            if w < 0:
                raise ValueError(f"feature weight for '{name}' must be >= 0.")

    def classify_history_status(self, prior_observation_count: int) -> str:
        """Prior-only history quality for the event being scored."""
        if prior_observation_count <= 0:
            return HISTORY_NONE
        if prior_observation_count < self.min_observations_for_limited_history:
            return HISTORY_INSUFFICIENT
        if prior_observation_count < self.min_observations_for_established_baseline:
            return HISTORY_LIMITED
        return HISTORY_ESTABLISHED

    def classify_anomaly_status(self, score: float | None, history_status: str) -> str:
        """Map score + history quality to anomaly_status."""
        if history_status in (HISTORY_NONE, HISTORY_INSUFFICIENT) or score is None:
            return INSUFFICIENT_HISTORY
        if score < self.normal_max_score:
            return NORMAL
        if score < self.elevated_max_score:
            return ELEVATED
        return ANOMALOUS

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["events_path"] = str(d["events_path"])
        d["fingerprints_path"] = str(d["fingerprints_path"])
        return d

    def describe_rationale(self) -> dict[str, str]:
        return {
            "history_thresholds": (
                "Same engineering cutoffs as Stage I.3 (3 for limited, 10 for "
                "established). Not scientifically validated sample sizes."
            ),
            "score_thresholds": (
                f"NORMAL if score < {self.normal_max_score}; ELEVATED if "
                f"[{self.normal_max_score}, {self.elevated_max_score}); "
                f"ANOMALOUS if >= {self.elevated_max_score}. Scale is a robust "
                "deviation index (median/MAD units), not a z-score and not a "
                "probability."
            ),
            "feature_weights": (
                "Interpretable relative importance for the weighted mean of "
                "available feature deviations. Not claimed to be optimal."
            ),
            "zero_mad_constant_mismatch_deviation": (
                f"When historical MAD is 0 and the current value differs from a "
                f"constant baseline, assign {self.zero_mad_constant_mismatch_deviation} "
                "rather than dividing by an undocumented epsilon. Documents "
                "'differs from historically constant behaviour'."
            ),
            "walk_forward": (
                "Each event is scored only against prior confirmed associations "
                "at the same facility. The current event is never in its own baseline."
            ),
        }


DEFAULT_CONFIG = AnomalyConfig()
