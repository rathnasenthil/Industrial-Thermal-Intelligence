"""JSON report assembly for GIFT Stage I.7."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evidence_fusion.config import EvidenceFusionConfig


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else round(float(value), 6)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def build_fusion_report(
    *,
    config: EvidenceFusionConfig,
    events_input_path: str,
    output_path: str,
    event_count: int,
    output_df: pd.DataFrame,
    processing_seconds: float,
    warnings: list[str] | None = None,
    domain_availability: dict[str, bool] | None = None,
    i4_columns_present: list[str] | None = None,
    i5_columns_present: list[str] | None = None,
) -> dict[str, Any]:
    candidate_counts = (
        output_df["source_intelligence_candidate"].value_counts(dropna=False).to_dict()
    )
    sufficiency_counts = output_df["evidence_sufficiency"].value_counts(dropna=False).to_dict()
    uncertainty_counts = output_df["evidence_uncertainty"].value_counts(dropna=False).to_dict()
    conflict_count = int(output_df["evidence_conflict_flag"].fillna(False).astype(bool).sum())

    null_counts = {
        "candidate_rationale_null": int(output_df["candidate_rationale"].isna().sum()),
        "evidence_profile_codes_empty": int(
            (output_df["evidence_profile_codes"].fillna("").astype(str) == "").sum()
        ),
        "sta_association_signal_unavailable": int(
            (output_df["sta_association_signal"] == "UNAVAILABLE").sum()
        ),
        "environmental_domain_unavailable": int(
            (~output_df["environmental_domain_available"].fillna(False).astype(bool)).sum()
        ),
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "GIFT Stage I.7 - Evidence Fusion / Source Intelligence",
        "input_file": events_input_path,
        "output_file": output_path,
        "event_count": int(event_count),
        "unique_event_id_count": int(output_df["event_id"].nunique()),
        "domain_availability": domain_availability or {},
        "candidate_distribution": {str(k): int(v) for k, v in candidate_counts.items()},
        "evidence_sufficiency_distribution": {
            str(k): int(v) for k, v in sufficiency_counts.items()
        },
        "evidence_uncertainty_distribution": {
            str(k): int(v) for k, v in uncertainty_counts.items()
        },
        "conflict_statistics": {
            "events_with_conflicts": conflict_count,
            "events_without_conflicts": int(event_count) - conflict_count,
        },
        "evidence_score_distributions": {
            "infrastructure_evidence_score": {
                str(k): int(v)
                for k, v in output_df["infrastructure_evidence_score"].value_counts().sort_index().items()
            },
            "temporal_evidence_score": {
                str(k): int(v)
                for k, v in output_df["temporal_evidence_score"].value_counts().sort_index().items()
            },
            "historical_evidence_score": {
                str(k): int(v)
                for k, v in output_df["historical_evidence_score"].value_counts().sort_index().items()
            },
            "anomaly_evidence_score": {
                str(k): int(v)
                for k, v in output_df["anomaly_evidence_score"].value_counts().sort_index().items()
            },
            "industrial_evidence_score": {
                str(k): int(v)
                for k, v in output_df["industrial_evidence_score"].value_counts().sort_index().items()
            },
            "evidence_strength": {
                str(k): int(v) for k, v in output_df["evidence_strength"].value_counts().items()
            },
        },
        "null_unavailable_counts": null_counts,
        "candidate_is_ground_truth_all_false": bool(
            output_df["candidate_is_ground_truth"].eq(False).all()
        ),
        "immutability": {
            "i4_columns_present": i4_columns_present or [],
            "i5_columns_present": i5_columns_present or [],
            "notes": (
                "I.7 appends fusion columns only. Prior-stage fields are never "
                "recalculated. Candidates are not ground truth."
            ),
        },
        "processing_time_seconds": round(processing_seconds, 3),
        "warnings": warnings or [],
        "limitations": [
            "Evidence fusion is not ground truth generation.",
            "I.7 candidate interpretations are deterministic evidence-based interpretations, not independently validated source labels.",
            "Ordinal evidence scores are engineering support levels, not probabilities.",
            "ANOMALOUS != INDUSTRIAL_FIRE.",
            "PERSISTENT != INDUSTRIAL_FIRE.",
            "OSM facility association != source classification.",
            "Facility proximity != proof of industrial origin.",
            "STA support != ground truth; NO_STA_ASSOCIATION != NOT_INDUSTRIAL.",
            "Environmental context != source label; missing env != negative evidence.",
            "NO_FACILITY_ASSOCIATION != NATURAL.",
            "No machine learning, pseudo-labels, risk scores, or scientific validation claims.",
            "Fusion thresholds and weights are engineering defaults.",
        ],
        "configuration": config.to_dict(),
        "rationale": config.describe_rationale(),
    }
    return _to_jsonable(report)


def save_report(report: dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return out
