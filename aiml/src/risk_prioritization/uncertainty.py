"""Uncertainty / data-quality component for Stage VI.

Missing evidence is UNAVAILABLE, not negative industrial evidence.
Uncertainty is reported separately from industrial_context and risk drivers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk_prioritization.config import (
    UNCERTAINTY_HIGH,
    UNCERTAINTY_LOW,
    UNCERTAINTY_MODERATE,
    UNCERTAINTY_VERY_HIGH,
    RiskPrioritizationConfig,
)


def _truthy(val) -> bool:
    if val is None:
        return False
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, float) and val != val:
        return False
    return str(val).strip().lower() in {"true", "1", "yes"}


def compute_uncertainty(
    events: pd.DataFrame,
    config: RiskPrioritizationConfig,
) -> pd.DataFrame:
    """uncertainty_score on 0-100 (higher = more uncertain)."""
    _ = config
    n = len(events)
    scores = np.zeros(n, dtype=float)
    bands = np.empty(n, dtype=object)
    factors = np.empty(n, dtype=object)
    limiting = np.empty(n, dtype=object)

    methods = events["facility_association_method"].astype(str).to_numpy() if "facility_association_method" in events.columns else np.full(n, "")
    confs = events["facility_attribution_confidence"].astype(str).to_numpy() if "facility_attribution_confidence" in events.columns else np.full(n, "")
    hist = events["baseline_history_status"].astype(str).to_numpy() if "baseline_history_status" in events.columns else np.full(n, "")
    persist = events["persistence_label"].astype(str).to_numpy() if "persistence_label" in events.columns else np.full(n, "")
    anomaly = events["anomaly_status"].astype(str).to_numpy() if "anomaly_status" in events.columns else np.full(n, "")
    sta_ok = events["sta_domain_available"].to_numpy() if "sta_domain_available" in events.columns else np.full(n, False)
    env_ok = events["environmental_domain_available"].to_numpy() if "environmental_domain_available" in events.columns else np.full(n, False)
    existing_limiting = (
        events["limiting_evidence_codes"].astype(str).to_numpy()
        if "limiting_evidence_codes" in events.columns
        else np.full(n, "", dtype=object)
    )

    for i in range(n):
        pts = 0.0
        fac: list[str] = []
        lim: list[str] = []

        if methods[i] == "NO_FACILITY_ASSOCIATION":
            pts += 25
            fac.append("NO_FACILITY_ASSOCIATION")
            lim.append("NO_FACILITY_ASSOCIATION")
        elif methods[i] == "AMBIGUOUS":
            pts += 30
            fac.append("AMBIGUOUS_FACILITY_ASSOCIATION")
            lim.append("AMBIGUOUS_FACILITY_ASSOCIATION")
        elif confs[i] == "LOW":
            pts += 12
            fac.append("LOW_FACILITY_ATTRIBUTION_CONFIDENCE")

        if hist[i] in ("NO_PRIOR_OBSERVATIONS", "INSUFFICIENT_HISTORY", "NOT_APPLICABLE"):
            pts += 15
            fac.append("INSUFFICIENT_OR_NO_FACILITY_HISTORY")
            lim.append("INSUFFICIENT_HISTORY")

        if persist[i] == "INSUFFICIENT_OBSERVATIONS":
            pts += 10
            fac.append("INSUFFICIENT_OBSERVATIONS")

        if anomaly[i] == "INSUFFICIENT_HISTORY":
            pts += 10
            fac.append("NO_ANOMALY_BASELINE")

        if not _truthy(sta_ok[i]):
            # Unavailable — recorded, contributes to uncertainty visibility, NOT anti-industrial.
            pts += 8
            fac.append("STA_UNAVAILABLE")
            lim.append("STA_UNAVAILABLE")

        if not _truthy(env_ok[i]):
            pts += 8
            fac.append("ENVIRONMENTAL_CONTEXT_UNAVAILABLE")
            lim.append("ENVIRONMENTAL_CONTEXT_UNAVAILABLE")

        # Preserve upstream limiting codes (dedupe)
        for code in str(existing_limiting[i] or "").split(";"):
            code = code.strip()
            if code and code not in lim:
                lim.append(code)

        pts = min(100.0, pts)
        scores[i] = round(pts, 4)
        if pts >= 70:
            bands[i] = UNCERTAINTY_VERY_HIGH
        elif pts >= 45:
            bands[i] = UNCERTAINTY_HIGH
        elif pts >= 25:
            bands[i] = UNCERTAINTY_MODERATE
        else:
            bands[i] = UNCERTAINTY_LOW
        factors[i] = ";".join(fac)
        limiting[i] = ";".join(lim)

    return pd.DataFrame(
        {
            "event_id": events["event_id"].astype(str).to_numpy(),
            "uncertainty_score": scores,
            "uncertainty_band": bands,
            "dominant_uncertainty_factors": factors,
            "risk_limiting_evidence_codes": limiting,
        }
    )
