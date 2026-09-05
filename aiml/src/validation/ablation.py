"""Ablation analysis over available I.7 evidence families (no fabricated STA/env)."""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from src.evidence_fusion.evidence_scores import (
    aggregate_industrial_score,
    corroboration_score,
    score_anomaly,
    score_historical,
    score_infrastructure,
    score_temporal,
)
from src.validation.metrics import evaluate_binary

# Candidate string literals (evaluation mapping only)
_INDUSTRIAL = "INDUSTRIAL_ACTIVITY_CANDIDATE"
_POSSIBLE = "POSSIBLE_INDUSTRIAL_ACTIVITY"
_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
_AMBIGUOUS = "AMBIGUOUS_EVIDENCE"


def _candidate_from_scores(method: str, infra: int, temporal: int, historical: int, anomaly: int) -> str:
    """Simplified candidate gate mirroring corrected I.7 rules (infra/temporal/history/anomaly)."""
    if method == "AMBIGUOUS":
        return _AMBIGUOUS
    industrial, t_eff, h_eff, a_eff = aggregate_industrial_score(infra, temporal, historical, anomaly, 0)
    corr = corroboration_score(t_eff, h_eff, a_eff, 0)
    if infra >= 3:
        return _INDUSTRIAL
    if infra >= 2 and corr >= 1:
        return _POSSIBLE
    if infra >= 1 and corr >= 2:
        return _POSSIBLE
    return _INSUFFICIENT


def _family_scores(row: pd.Series, *, use_temporal: bool, use_history: bool, use_anomaly: bool) -> tuple[int, int, int, int]:
    method = str(row.get("facility_association_method", "NO_FACILITY_ASSOCIATION"))
    conf = str(row.get("facility_attribution_confidence", "NONE"))
    ftype = row.get("facility_type")
    ftype_s = None if pd.isna(ftype) else str(ftype)
    infra = score_infrastructure(method, conf, ftype_s)
    temporal = score_temporal(str(row.get("persistence_label", ""))) if use_temporal else 0
    historical = score_historical(str(row.get("baseline_history_status", "")), method) if use_history else 0
    anomaly = score_anomaly(str(row.get("anomaly_status", ""))) if use_anomaly else 0
    return infra, temporal, historical, anomaly


def run_ablation(
    matched: pd.DataFrame,
    events: pd.DataFrame,
    *,
    y_true_col: str = "reference_label_normalized",
) -> dict[str, Any]:
    """Compare family subsets on matched independent labels.

    STA/environmental families are reported unavailable when not present —
    never fabricated.
    """
    if matched.empty:
        return {
            "metric_status": "NOT_EVALUATED",
            "reason": "No matched independent validation records for ablation.",
            "variants": {},
            "unavailable_families": ["sta", "environmental"],
        }

    # Join event evidence columns needed for rescoring
    need = [
        "event_id",
        "facility_association_method",
        "facility_attribution_confidence",
        "facility_type",
        "persistence_label",
        "baseline_history_status",
        "anomaly_status",
    ]
    have = [c for c in need if c in events.columns]
    ev = events[have].drop_duplicates("event_id")
    df = matched.merge(ev, on="event_id", how="left", suffixes=("", "_ev"))
    df = df[df["validation_match_status"] == "MATCHED"].copy()
    df = df[df["validation_source_independent"].fillna(False).astype(bool)].copy()
    if df.empty:
        return {
            "metric_status": "NOT_EVALUATED",
            "reason": "No independent MATCHED records for ablation.",
            "variants": {},
            "unavailable_families": ["sta", "environmental"],
        }

    variants = {
        "infrastructure_only": (False, False, False),
        "infrastructure_plus_temporal": (True, False, False),
        "infrastructure_plus_history": (False, True, False),
        "infrastructure_plus_anomaly": (False, False, True),
        "infrastructure_plus_temporal_history_anomaly": (True, True, True),
    }
    results: dict[str, Any] = {}
    for name, (use_t, use_h, use_a) in variants.items():
        preds = []
        truths = []
        for _, row in df.iterrows():
            method = str(row.get("facility_association_method", "NO_FACILITY_ASSOCIATION"))
            infra, temporal, historical, anomaly = _family_scores(
                row, use_temporal=use_t, use_history=use_h, use_anomaly=use_a
            )
            preds.append(_candidate_from_scores(method, infra, temporal, historical, anomaly))
            truths.append(str(row[y_true_col]))
        metrics = evaluate_binary(truths, preds, mode="strict")
        results[name] = metrics

    # Full I.7 uses already-produced candidate on the match row
    full = evaluate_binary(
        df[y_true_col].astype(str).tolist(),
        df["source_intelligence_candidate"].tolist(),
        mode="strict",
    )
    results["full_available_i7"] = full

    return {
        "metric_status": "EVALUATED",
        "variants": results,
        "unavailable_families": {
            "sta": "NOT_EVALUATED — STA evidence unavailable in current production inputs",
            "environmental": "NOT_EVALUATED — environmental context unavailable in current production inputs",
        },
        "note": (
            "Ablation results are empirical under the tested independent dataset; "
            "they do not establish causal necessity of each family."
        ),
    }
