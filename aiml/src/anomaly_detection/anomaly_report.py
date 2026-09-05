"""JSON report assembly for GIFT Stage I.4."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.anomaly_detection.config import AnomalyConfig


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
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    return value


def build_anomaly_report(
    *,
    config: AnomalyConfig,
    events_input_path: str,
    fingerprints_input_path: str,
    output_df: pd.DataFrame,
    processing_seconds: float,
) -> dict[str, Any]:
    """Assemble the Stage I.4 report."""
    total = len(output_df)
    methods = output_df["facility_association_method"].value_counts()
    confirmed = int(output_df["facility_id"].notna().sum())
    ambiguous = int(methods.get("AMBIGUOUS", 0))
    no_assoc = int(methods.get("NO_FACILITY_ASSOCIATION", 0))

    status_counts = output_df["anomaly_status"].value_counts()
    confidence_counts = output_df["anomaly_confidence"].value_counts()
    history_counts = output_df["baseline_history_status"].value_counts()

    feature_cols = [
        "peak_frp_deviation",
        "event_size_deviation",
        "duration_deviation",
        "distance_deviation",
        "persistence_deviation",
        "monthly_deviation",
    ]
    feature_availability = {}
    for col in feature_cols:
        available = int(output_df[col].notna().sum())
        feature_availability[col] = {
            "available_count": available,
            "missing_count": int(total - available),
        }

    scored = output_df.loc[output_df["anomaly_score"].notna(), "anomaly_score"]
    score_stats = (
        {
            "min": float(scored.min()),
            "median": float(scored.median()),
            "mean": round(float(scored.mean()), 4),
            "max": float(scored.max()),
            "count": int(len(scored)),
        }
        if not scored.empty
        else {"min": None, "median": None, "mean": None, "max": None, "count": 0}
    )

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_stage": "GIFT Stage I.4 - Temporal Deviation & Anomaly Detection",
        "input": {
            "events_input_path": events_input_path,
            "fingerprints_input_path": fingerprints_input_path,
            "total_events_processed": int(total),
            "events_with_confirmed_facility_association": confirmed,
            "ambiguous_events": ambiguous,
            "events_without_facility_association": no_assoc,
        },
        "history_at_scoring_time": {
            "NO_PRIOR_OBSERVATIONS": int(history_counts.get("NO_PRIOR_OBSERVATIONS", 0)),
            "INSUFFICIENT_HISTORY": int(history_counts.get("INSUFFICIENT_HISTORY", 0)),
            "LIMITED_HISTORY": int(history_counts.get("LIMITED_HISTORY", 0)),
            "ESTABLISHED_BASELINE": int(history_counts.get("ESTABLISHED_BASELINE", 0)),
            "NOT_APPLICABLE": int(history_counts.get("NOT_APPLICABLE", 0)),
        },
        "anomaly_status_counts": {
            "INSUFFICIENT_HISTORY": int(status_counts.get("INSUFFICIENT_HISTORY", 0)),
            "NORMAL": int(status_counts.get("NORMAL", 0)),
            "ELEVATED": int(status_counts.get("ELEVATED", 0)),
            "ANOMALOUS": int(status_counts.get("ANOMALOUS", 0)),
        },
        "anomaly_confidence_counts": {
            "NONE": int(confidence_counts.get("NONE", 0)),
            "LOW": int(confidence_counts.get("LOW", 0)),
            "MEDIUM": int(confidence_counts.get("MEDIUM", 0)),
            "HIGH": int(confidence_counts.get("HIGH", 0)),
        },
        "anomaly_score_statistics_among_scored_events": score_stats,
        "feature_availability": feature_availability,
        "configuration": {
            **config.to_dict(),
            "rationale": config.describe_rationale(),
        },
        "leakage_validation": {
            "walk_forward_prior_only": True,
            "notes": (
                "Each confirmed associated event is scored using only earlier "
                "confirmed associations at the same facility (sorted by "
                "event_start, then event_id). The current event is appended to "
                "facility history only AFTER scoring. AMBIGUOUS and "
                "NO_FACILITY_ASSOCIATION events never enter any facility baseline. "
                "I.3 full-history fingerprints are not used as the scoring baseline "
                "(that would leak future observations)."
            ),
        },
        "performance": {"processing_seconds": round(processing_seconds, 3)},
        "reproducibility": {
            "deterministic": True,
            "notes": (
                "Identical inputs and configuration produce identical outputs. "
                "Ordering uses mergesort on (event_start, event_id); no random "
                "seeds; no unordered set iteration affecting results."
            ),
        },
        "limitations": [
            "I.4 detects unusual thermal behaviour relative to a facility's "
            "prior confirmed associations. It does NOT classify the source as "
            "an industrial fire, wildfire, agricultural burn, or any other cause.",
            "Most OSM facilities have no confirmed thermal observations; many "
            "events therefore receive INSUFFICIENT_HISTORY.",
            "Facility association (Stage I.2) is contextual spatial evidence, "
            "not proof of source identity.",
            "FIRMS observations are discrete satellite overpasses, not continuous "
            "ground truth.",
            "Stage G.1 persistence labels describe observed detection patterns, "
            "not physical fire persistence; I.4 consumes them without redefining them.",
            "Long-running Stage G events are not artificially split to improve "
            "anomaly statistics.",
            "The 2023–2024 historical window may not capture all legitimate "
            "operational regimes for every facility.",
            "Anomaly thresholds and feature weights are engineering choices and "
            "have not been scientifically validated. Do not interpret anomaly_score "
            "as a risk score or fire probability.",
            "No accuracy/precision/recall is reported — no independent ground-truth "
            "validation set is used in this stage.",
            "Monthly baselines are prior-only and require a minimum number of prior "
            "same-month observations; otherwise monthly_deviation is null.",
            "When historical MAD is very small but non-zero, the robust "
            "deviation index |x-median|/MAD can become extremely large for "
            "features such as duration on long-running persistent events; "
            "status thresholds still apply, but raw score magnitudes should "
            "be interpreted cautiously alongside feature-level contributions.",
        ],
    }
    return _to_jsonable(report)


def save_report(report: dict[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=False)
