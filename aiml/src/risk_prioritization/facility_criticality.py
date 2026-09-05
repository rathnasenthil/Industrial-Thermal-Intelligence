"""Facility / infrastructure context component (I.2).

Facility proximity != industrial source proof.
Facility type weights are engineering context, not hazard claims.
NO_FACILITY / AMBIGUOUS remain in the output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk_prioritization.config import RiskPrioritizationConfig

_RELATION_LEVEL: dict[str, float] = {
    "WITHIN_FACILITY": 1.0,
    "INTERSECTS_FACILITY": 0.95,
    "NEAR_FACILITY": 0.45,
    "AMBIGUOUS": 0.20,
    "NO_FACILITY_ASSOCIATION": 0.0,
}

_CONF_LEVEL: dict[str, float] = {
    "HIGH": 1.0,
    "MEDIUM": 0.75,
    "LOW": 0.45,
    "NONE": 0.0,
}


def compute_facility_context(
    events: pd.DataFrame,
    config: RiskPrioritizationConfig,
) -> pd.DataFrame:
    n = len(events)
    methods = (
        events["facility_association_method"].astype(str).to_numpy()
        if "facility_association_method" in events.columns
        else np.full(n, "NO_FACILITY_ASSOCIATION", dtype=object)
    )
    confs = (
        events["facility_attribution_confidence"].astype(str).to_numpy()
        if "facility_attribution_confidence" in events.columns
        else np.full(n, "NONE", dtype=object)
    )
    types = (
        events["facility_type"].to_numpy()
        if "facility_type" in events.columns
        else np.full(n, None)
    )

    scores = np.zeros(n, dtype=float)
    reasons = np.empty(n, dtype=object)
    for i in range(n):
        method = methods[i]
        conf = confs[i]
        ftype = None if types[i] is None or (isinstance(types[i], float) and types[i] != types[i]) else str(types[i])
        rel = _RELATION_LEVEL.get(method, 0.0)
        clev = _CONF_LEVEL.get(conf, 0.0)
        tw = float(config.facility_type_weights.get(ftype, 0.35)) if ftype else 0.0
        # Combine without letting type alone dominate: relation * (0.6 + 0.4*type) * confidence blend
        if method == "NO_FACILITY_ASSOCIATION":
            level = 0.0
            reason = "NO_FACILITY_ASSOCIATION"
        elif method == "AMBIGUOUS":
            level = 0.20
            reason = "AMBIGUOUS_FACILITY_ASSOCIATION"
        else:
            level = rel * (0.55 + 0.45 * tw) * (0.4 + 0.6 * clev)
            level = min(1.0, max(0.0, level))
            reason = f"FACILITY_{method}"
            if ftype:
                reason += f";TYPE_{ftype}"
            reason += f";CONF_{conf}"
        scores[i] = round(level * float(config.weight_facility), 4)
        reasons[i] = reason

    return pd.DataFrame(
        {
            "event_id": events["event_id"].astype(str).to_numpy(),
            "facility_context_score": scores,
            "facility_context_reason": reasons,
        }
    )
