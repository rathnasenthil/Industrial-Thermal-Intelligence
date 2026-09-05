"""Aggregate investigation priority scoring for Stage VI.

risk_score is a deterministic decision-support score on 0-100.
It is NOT a probability of industrial fire.
industrial_context is computed separately from investigation_priority.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk_prioritization.config import (
    ACTION_MONITOR,
    ACTION_PRIORITIZE,
    ACTION_REVIEW,
    ACTION_URGENT,
    INDUSTRIAL_CONTEXT_AMBIGUOUS,
    INDUSTRIAL_CONTEXT_INSUFFICIENT,
    INDUSTRIAL_CONTEXT_POSSIBLE,
    INDUSTRIAL_CONTEXT_STRONG,
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    SEVERITY_EXTREME,
    RiskPrioritizationConfig,
)


def map_industrial_context(candidate: str | None) -> str:
    """Map I.7 candidate interpretation to industrial_context (separate from priority)."""
    c = str(candidate or "")
    if c == "INDUSTRIAL_ACTIVITY_CANDIDATE":
        return INDUSTRIAL_CONTEXT_STRONG
    if c == "POSSIBLE_INDUSTRIAL_ACTIVITY":
        return INDUSTRIAL_CONTEXT_POSSIBLE
    if c in {"AMBIGUOUS_EVIDENCE", "MIXED_OR_CONFLICTING"}:
        return INDUSTRIAL_CONTEXT_AMBIGUOUS
    return INDUSTRIAL_CONTEXT_INSUFFICIENT


def industrial_evidence_component(
    industrial_evidence_score: float | None,
    config: RiskPrioritizationConfig,
) -> float:
    """Scale I.7 industrial_evidence_score (0..14 documented) into component points."""
    if industrial_evidence_score is None or (
        isinstance(industrial_evidence_score, float) and industrial_evidence_score != industrial_evidence_score
    ):
        return 0.0
    # Documented I.7 max = 14
    frac = min(max(float(industrial_evidence_score) / 14.0, 0.0), 1.0)
    return round(frac * float(config.weight_industrial_evidence), 4)


def priority_from_score(score: float, config: RiskPrioritizationConfig) -> str:
    if score >= config.priority_critical_min:
        return PRIORITY_CRITICAL
    if score >= config.priority_high_min:
        return PRIORITY_HIGH
    if score >= config.priority_medium_min:
        return PRIORITY_MEDIUM
    return PRIORITY_LOW


def action_from_priority(priority: str) -> str:
    return {
        PRIORITY_LOW: ACTION_MONITOR,
        PRIORITY_MEDIUM: ACTION_REVIEW,
        PRIORITY_HIGH: ACTION_PRIORITIZE,
        PRIORITY_CRITICAL: ACTION_URGENT,
    }.get(priority, ACTION_MONITOR)


_PRIORITY_RANK = {
    PRIORITY_LOW: 0,
    PRIORITY_MEDIUM: 1,
    PRIORITY_HIGH: 2,
    PRIORITY_CRITICAL: 3,
}


def aggregate_priority(
    events: pd.DataFrame,
    thermal: pd.DataFrame,
    persistence: pd.DataFrame,
    anomaly: pd.DataFrame,
    facility: pd.DataFrame,
    uncertainty: pd.DataFrame,
    config: RiskPrioritizationConfig,
) -> pd.DataFrame:
    """Combine components into risk_score / investigation_priority / industrial_context."""
    event_ids = events["event_id"].astype(str)
    t = thermal.set_index("event_id").reindex(event_ids)
    p = persistence.set_index("event_id").reindex(event_ids)
    a = anomaly.set_index("event_id").reindex(event_ids)
    f = facility.set_index("event_id").reindex(event_ids)
    u = uncertainty.set_index("event_id").reindex(event_ids)

    n = len(events)
    ie_raw = (
        pd.to_numeric(events["industrial_evidence_score"], errors="coerce").to_numpy()
        if "industrial_evidence_score" in events.columns
        else np.full(n, np.nan)
    )
    candidates = (
        events["source_intelligence_candidate"].astype(str).to_numpy()
        if "source_intelligence_candidate" in events.columns
        else np.full(n, "", dtype=object)
    )
    methods = (
        events["facility_association_method"].astype(str).to_numpy()
        if "facility_association_method" in events.columns
        else np.full(n, "", dtype=object)
    )

    risk_scores = np.zeros(n, dtype=float)
    ie_comp = np.zeros(n, dtype=float)
    contexts = np.empty(n, dtype=object)
    priorities = np.empty(n, dtype=object)
    actions = np.empty(n, dtype=object)

    for i in range(n):
        eid = event_ids.iloc[i]
        thermal_s = float(t.at[eid, "thermal_severity_score"] or 0.0)
        persist_s = float(p.at[eid, "persistence_priority_score"] or 0.0)
        anomaly_s = float(a.at[eid, "anomaly_priority_score"] or 0.0)
        facility_s = float(f.at[eid, "facility_context_score"] or 0.0)
        ie = industrial_evidence_component(ie_raw[i], config)
        ie_comp[i] = ie

        positive = thermal_s + persist_s + anomaly_s + facility_s + ie
        # Ambiguity dampening only (NOT missing STA/env)
        damp = 0.0
        if methods[i] == "AMBIGUOUS":
            damp = min(float(config.ambiguity_dampening_max), positive * 0.12)

        score = max(0.0, min(100.0, positive - damp))
        risk_scores[i] = round(score, 4)

        contexts[i] = map_industrial_context(candidates[i])
        pr = priority_from_score(score, config)
        # Documented floor: EXTREME thermal alone → at least MEDIUM
        if str(t.at[eid, "thermal_severity_band"]) == SEVERITY_EXTREME:
            if _PRIORITY_RANK[pr] < _PRIORITY_RANK[config.extreme_thermal_minimum_priority]:
                pr = config.extreme_thermal_minimum_priority
        priorities[i] = pr
        actions[i] = action_from_priority(pr)

    return pd.DataFrame(
        {
            "event_id": event_ids.to_numpy(),
            "risk_score": risk_scores,
            "investigation_priority": priorities,
            "recommended_action": actions,
            "industrial_context": contexts,
            "industrial_evidence_component": ie_comp,
            "thermal_severity_score": t["thermal_severity_score"].to_numpy(),
            "thermal_severity_band": t["thermal_severity_band"].to_numpy(),
            "persistence_priority_score": p["persistence_priority_score"].to_numpy(),
            "persistence_priority_reason": p["persistence_priority_reason"].to_numpy(),
            "anomaly_priority_score": a["anomaly_priority_score"].to_numpy(),
            "anomaly_priority_reason": a["anomaly_priority_reason"].to_numpy(),
            "facility_context_score": f["facility_context_score"].to_numpy(),
            "facility_context_reason": f["facility_context_reason"].to_numpy(),
            "uncertainty_score": u["uncertainty_score"].to_numpy(),
            "uncertainty_band": u["uncertainty_band"].to_numpy(),
            "dominant_uncertainty_factors": u["dominant_uncertainty_factors"].to_numpy(),
            "risk_limiting_evidence_codes": u["risk_limiting_evidence_codes"].to_numpy(),
        }
    )
