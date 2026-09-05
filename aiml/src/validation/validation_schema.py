"""Canonical independent-validation schema and helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

CANONICAL_COLUMNS: tuple[str, ...] = (
    "validation_id",
    "event_id",  # optional pre-linked event; may be null before matching
    "reference_label_raw",
    "reference_label_normalized",
    "reference_source",
    "reference_date",
    "reference_latitude",
    "reference_longitude",
    "reference_geometry_wkt",
    "reference_confidence",
    "label_notes",
    "validation_source",
    "validation_source_independent",
    "validation_label_verified",
    "validation_match_status",
)

MATCH_OUTPUT_COLUMNS: tuple[str, ...] = (
    "validation_id",
    "event_id",
    "reference_label_raw",
    "reference_label_normalized",
    "reference_source",
    "reference_date",
    "reference_latitude",
    "reference_longitude",
    "validation_source_independent",
    "validation_match_status",
    "match_distance_km",
    "match_time_delta_hours",
    "candidate_match_count",
    "source_intelligence_candidate",
    "evidence_strength",
    "industrial_evidence_score",
)


def empty_canonical_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(CANONICAL_COLUMNS))


def empty_matches_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(MATCH_OUTPUT_COLUMNS))


def clean_text(value: Any, default: str | None = None) -> str | None:
    if value is None:
        return default
    if isinstance(value, float) and value != value:
        return default
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return default
    return text


def not_evaluated_block(reason: str) -> dict[str, Any]:
    return {
        "metric_status": "NOT_EVALUATED",
        "reason": reason,
        "precision": None,
        "recall": None,
        "f1": None,
        "specificity": None,
        "balanced_accuracy": None,
        "accuracy": None,
        "ppv": None,
        "npv": None,
        "confusion_matrix": None,
        "sample_count": 0,
    }
