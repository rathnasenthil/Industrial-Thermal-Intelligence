"""
Configuration for GIFT Stage V (Independent Validation & Evaluation).

Spatial/temporal match tolerances are ENGINEERING defaults — not
scientifically validated optima.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

STATUS_UNAVAILABLE = "VALIDATION_DATA_UNAVAILABLE"
STATUS_EVALUATED = "VALIDATION_EVALUATED"
STATUS_PARTIAL = "VALIDATION_PARTIAL"

METRIC_NOT_EVALUATED = "NOT_EVALUATED"

# Match status vocabulary
MATCHED = "MATCHED"
MULTIPLE_POSSIBLE_MATCHES = "MULTIPLE_POSSIBLE_MATCHES"
NO_EVENT_MATCH = "NO_EVENT_MATCH"
INVALID_REFERENCE = "INVALID_REFERENCE"

# Normalized reference labels
LABEL_INDUSTRIAL = "INDUSTRIAL"
LABEL_NATURAL = "NATURAL"
LABEL_AGRICULTURAL = "AGRICULTURAL"
LABEL_OTHER = "OTHER"
LABEL_AMBIGUOUS = "AMBIGUOUS"
LABEL_UNKNOWN = "UNKNOWN"

NORMALIZED_LABELS: tuple[str, ...] = (
    LABEL_INDUSTRIAL,
    LABEL_NATURAL,
    LABEL_AGRICULTURAL,
    LABEL_OTHER,
    LABEL_AMBIGUOUS,
    LABEL_UNKNOWN,
)

# I.7 candidate categories treated as abstentions under coverage-aware eval
ABSTENTION_CANDIDATES: frozenset[str] = frozenset(
    {
        "INSUFFICIENT_EVIDENCE",
        "AMBIGUOUS_EVIDENCE",
        "MIXED_OR_CONFLICTING",
    }
)

# Strict positive / negative mapping for binary industrial evaluation
STRICT_POSITIVE_CANDIDATES: frozenset[str] = frozenset({"INDUSTRIAL_ACTIVITY_CANDIDATE"})
STRICT_NEGATIVE_CANDIDATES: frozenset[str] = frozenset(
    {
        "ENVIRONMENTAL_VEGETATION_CONTEXT",
        "ENVIRONMENTAL_AGRICULTURE_CONTEXT",
    }
)
# POSSIBLE is evaluated as positive only in a documented "inclusive" mode
INCLUSIVE_POSITIVE_CANDIDATES: frozenset[str] = frozenset(
    {"INDUSTRIAL_ACTIVITY_CANDIDATE", "POSSIBLE_INDUSTRIAL_ACTIVITY"}
)

FORBIDDEN_PSEUDO_LABEL_SOURCES: frozenset[str] = frozenset(
    {
        "i2_facility_association",
        "facility_association_method",
        "i7_candidate",
        "source_intelligence_candidate",
        "sta_association",
        "anomaly_status",
        "persistence_label",
        "pipeline_derived",
        "circular",
    }
)


@dataclass(frozen=True)
class ValidationConfig:
    """Engineering parameters for Stage V independent validation."""

    events_path: Path = Path("data/processed/thermal_events_with_evidence_fusion.csv")
    validation_path: Path = Path("data/external/validation_labels.csv")
    validation_search_dirs: tuple[Path, ...] = field(
        default_factory=lambda: (
            Path("data/external"),
            Path("data/raw"),
            Path("data/processed"),
        )
    )
    validation_filename_hints: tuple[str, ...] = (
        "validation",
        "labels",
        "ground_truth",
        "ground-truth",
        "incident",
        "curated",
        "reference_labels",
    )
    spatial_tolerance_km: float = 5.0
    temporal_tolerance_hours: float = 72.0
    ambiguity_distance_tolerance_km: float = 0.5
    require_independent_source: bool = True
    min_subgroup_count: int = 30
    binary_positive_label: str = LABEL_INDUSTRIAL
    binary_negative_labels: tuple[str, ...] = (
        LABEL_NATURAL,
        LABEL_AGRICULTURAL,
        LABEL_OTHER,
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "events_path": str(self.events_path),
            "validation_path": str(self.validation_path),
            "validation_search_dirs": [str(p) for p in self.validation_search_dirs],
            "validation_filename_hints": list(self.validation_filename_hints),
            "spatial_tolerance_km": self.spatial_tolerance_km,
            "temporal_tolerance_hours": self.temporal_tolerance_hours,
            "ambiguity_distance_tolerance_km": self.ambiguity_distance_tolerance_km,
            "require_independent_source": self.require_independent_source,
            "min_subgroup_count": self.min_subgroup_count,
            "binary_positive_label": self.binary_positive_label,
            "binary_negative_labels": list(self.binary_negative_labels),
            "rationale": {
                "spatial_tolerance_km": (
                    "Engineering match radius (default 5 km) for associating an "
                    "independent reference point with a thermal event centroid — "
                    "not a scientifically validated fire-attribution distance."
                ),
                "temporal_tolerance_hours": (
                    "Engineering time window (default 72 h) around reference_date — "
                    "not an optimal detection-latency parameter."
                ),
                "independence": (
                    "Pipeline-derived evidence (I.2/I.3/I.4/I.5/I.6/I.7) is never "
                    "accepted as ground truth."
                ),
            },
        }
