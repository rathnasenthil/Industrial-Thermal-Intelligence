"""JSON report assembly for GIFT Stage V."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.validation.config import ValidationConfig


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


def build_validation_report(
    *,
    config: ValidationConfig,
    status: str,
    validation_dataset_available: bool,
    independent_validation_confirmed: bool,
    dataset_paths: list[str],
    dataset_audit: dict[str, Any],
    match_stats: dict[str, Any],
    leakage_audit: dict[str, Any],
    metrics: dict[str, Any],
    coverage_metrics: dict[str, Any],
    abstention_metrics: dict[str, Any],
    ablation_results: dict[str, Any],
    threshold_analysis: dict[str, Any],
    error_analysis: dict[str, Any],
    calibration_proxy: dict[str, Any],
    subgroup_analysis: dict[str, Any],
    processing_seconds: float,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "GIFT Stage V - Independent Validation & Evaluation",
        "status": status,
        "validation_dataset": dataset_paths,
        "validation_dataset_available": bool(validation_dataset_available),
        "independent_validation_confirmed": bool(independent_validation_confirmed),
        "reference_record_count": int(dataset_audit.get("total_reference_records", 0)),
        "evaluable_record_count": int(coverage_metrics.get("evaluable_record_count", 0)),
        "matched_event_count": int(match_stats.get("matched", 0)),
        "unmatched_reference_count": int(match_stats.get("no_event_match", 0)),
        "ambiguous_match_count": int(match_stats.get("multiple_possible_matches", 0)),
        "invalid_reference_count": int(match_stats.get("invalid_reference", 0)),
        "label_distribution": dataset_audit.get("label_distribution_normalized", {}),
        "date_coverage": dataset_audit.get("date_range"),
        "geographic_coverage": dataset_audit.get("geographic_coverage"),
        "dataset_audit": dataset_audit,
        "match_statistics": match_stats,
        "leakage_audit": leakage_audit,
        "metric_status": metrics.get("metric_status", "NOT_EVALUATED"),
        "classification_metrics": metrics,
        "coverage_metrics": coverage_metrics,
        "abstention_metrics": abstention_metrics,
        "ablation_results": ablation_results,
        "threshold_analysis": threshold_analysis,
        "error_analysis": error_analysis,
        "calibration_proxy": calibration_proxy,
        "subgroup_analysis": subgroup_analysis,
        "processing_time_seconds": round(processing_seconds, 3),
        "warnings": warnings or [],
        "limitations": [
            "Validation metrics are reported only when independent reference labels are available.",
            "Pipeline-derived evidence is not used as ground truth.",
            "I.7 candidate mappings used for evaluation are documented evaluation conventions, not identity with truth.",
            "Spatial/temporal match tolerances are engineering defaults.",
            "Ordinal evidence scores are not probabilities; no probability calibration is performed.",
            "Absence of independent validation data means NO VALIDATED PERFORMANCE CLAIM IS MADE.",
        ],
        "configuration": config.to_dict(),
    }
    if status == "VALIDATION_DATA_UNAVAILABLE":
        report["performance_claim"] = "NO VALIDATED PERFORMANCE CLAIM IS MADE."
    return _to_jsonable(report)


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return out
