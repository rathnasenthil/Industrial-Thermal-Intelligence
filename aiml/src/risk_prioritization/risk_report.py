"""JSON report for Stage VI risk prioritization."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.risk_prioritization.config import RiskPrioritizationConfig


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


def _dist(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}


def build_risk_report(
    *,
    config: RiskPrioritizationConfig,
    events_input_path: str,
    output_path: str,
    output_df: pd.DataFrame,
    processing_seconds: float,
    warnings: list[str] | None = None,
    sanity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sta_avail = False
    env_avail = False
    if "sta_domain_available" in output_df.columns:
        sta_avail = bool(output_df["sta_domain_available"].fillna(False).astype(bool).any())
    if "environmental_domain_available" in output_df.columns:
        env_avail = bool(output_df["environmental_domain_available"].fillna(False).astype(bool).any())

    # Dominant factor tallies
    dom_counts: dict[str, int] = {}
    for blob in output_df["dominant_risk_factors"].fillna("").astype(str):
        for part in blob.split(";"):
            part = part.strip()
            if part:
                dom_counts[part] = dom_counts.get(part, 0) + 1
    unc_counts: dict[str, int] = {}
    for blob in output_df["dominant_uncertainty_factors"].fillna("").astype(str):
        for part in blob.split(";"):
            part = part.strip()
            if part:
                unc_counts[part] = unc_counts.get(part, 0) + 1

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "GIFT Stage VI - Decision & Risk Prioritization",
        "version": config.scoring_version,
        "input_file": events_input_path,
        "output_file": output_path,
        "event_count": int(len(output_df)),
        "priority_distribution": _dist(output_df["investigation_priority"]),
        "risk_score_distribution": {
            "min": float(output_df["risk_score"].min()),
            "max": float(output_df["risk_score"].max()),
            "mean": float(output_df["risk_score"].mean()),
            "median": float(output_df["risk_score"].median()),
            "p25": float(output_df["risk_score"].quantile(0.25)),
            "p75": float(output_df["risk_score"].quantile(0.75)),
            "p95": float(output_df["risk_score"].quantile(0.95)),
        },
        "thermal_severity_distribution": _dist(output_df["thermal_severity_band"]),
        "uncertainty_distribution": _dist(output_df["uncertainty_band"]),
        "industrial_context_distribution": _dist(output_df["industrial_context"]),
        "component_score_distributions": {
            "thermal_severity_score": {
                "min": float(output_df["thermal_severity_score"].min()),
                "max": float(output_df["thermal_severity_score"].max()),
                "mean": float(output_df["thermal_severity_score"].mean()),
            },
            "persistence_priority_score": {
                "min": float(output_df["persistence_priority_score"].min()),
                "max": float(output_df["persistence_priority_score"].max()),
                "mean": float(output_df["persistence_priority_score"].mean()),
            },
            "anomaly_priority_score": {
                "min": float(output_df["anomaly_priority_score"].min()),
                "max": float(output_df["anomaly_priority_score"].max()),
                "mean": float(output_df["anomaly_priority_score"].mean()),
            },
            "facility_context_score": {
                "min": float(output_df["facility_context_score"].min()),
                "max": float(output_df["facility_context_score"].max()),
                "mean": float(output_df["facility_context_score"].mean()),
            },
            "industrial_evidence_component": {
                "min": float(output_df["industrial_evidence_component"].min()),
                "max": float(output_df["industrial_evidence_component"].max()),
                "mean": float(output_df["industrial_evidence_component"].mean()),
            },
        },
        "priority_by_industrial_context": (
            pd.crosstab(output_df["industrial_context"], output_df["investigation_priority"])
            .astype(int)
            .to_dict()
        ),
        "priority_by_uncertainty_band": (
            pd.crosstab(output_df["uncertainty_band"], output_df["investigation_priority"])
            .astype(int)
            .to_dict()
        ),
        "missing_evidence_summary": {
            "sta_unavailable_count": int((~output_df["sta_domain_available"].fillna(False).astype(bool)).sum())
            if "sta_domain_available" in output_df.columns
            else None,
            "environmental_unavailable_count": int(
                (~output_df["environmental_domain_available"].fillna(False).astype(bool)).sum()
            )
            if "environmental_domain_available" in output_df.columns
            else None,
        },
        "STA_availability": sta_avail,
        "environmental_availability": env_avail,
        "dominant_risk_factor_counts": dict(sorted(dom_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "dominant_uncertainty_factor_counts": dict(
            sorted(unc_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "sanity_checks": sanity or {},
        "processing_time_seconds": round(processing_seconds, 3),
        "warnings": warnings or [],
        "limitations": [
            "Investigation priority is a deterministic decision-support score, not a probability of industrial fire.",
            "Missing evidence is treated as unavailable evidence, not negative evidence.",
            "Risk prioritization has not been independently validated because Stage V currently has no independent reference dataset.",
            "Weights and thresholds are engineering defaults.",
            "ANOMALOUS != FIRE; PERSISTENT != industrial fire; facility proximity != industrial source proof.",
            "industrial_context is separate from investigation_priority.",
        ],
        "configuration": config.to_dict(),
    }
    return _to_jsonable(report)


def save_report(report: dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return out
