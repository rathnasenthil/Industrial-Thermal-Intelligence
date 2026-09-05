"""Temporal anomaly priority component (I.4).

ANOMALOUS != FIRE and ANOMALOUS != INDUSTRIAL.
Missing/insufficient history → score unavailable semantics (0 contribution,
explicit reason) — not fabricated anomaly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk_prioritization.config import RiskPrioritizationConfig

_ANOMALY_LEVEL: dict[str, tuple[float, str]] = {
    "ANOMALOUS": (1.0, "ANOMALOUS_TEMPORAL_BEHAVIOUR"),
    "ELEVATED": (0.55, "ELEVATED_TEMPORAL_DEVIATION"),
    "NORMAL": (0.10, "NORMAL_TEMPORAL_BEHAVIOUR"),
    "INSUFFICIENT_HISTORY": (0.0, "INSUFFICIENT_HISTORY_NO_ANOMALY_BASELINE"),
}


def compute_anomaly_priority(
    events: pd.DataFrame,
    config: RiskPrioritizationConfig,
) -> pd.DataFrame:
    n = len(events)
    status = (
        events["anomaly_status"].astype(str).to_numpy()
        if "anomaly_status" in events.columns
        else np.full(n, "INSUFFICIENT_HISTORY", dtype=object)
    )
    # Optional mild boost from anomaly_score when available (capped)
    raw_score = (
        pd.to_numeric(events["anomaly_score"], errors="coerce").to_numpy(dtype=float)
        if "anomaly_score" in events.columns
        else np.full(n, np.nan)
    )

    scores = np.zeros(n, dtype=float)
    reasons = np.empty(n, dtype=object)
    for i, st in enumerate(status):
        level, reason = _ANOMALY_LEVEL.get(st, (0.0, "ANOMALY_STATUS_UNKNOWN"))
        # If continuous score available and status is ELEVATED/ANOMALOUS, blend slightly
        if st in ("ANOMALOUS", "ELEVATED") and np.isfinite(raw_score[i]):
            # anomaly_score typically ~0-5+ in I.4; map softly into 0-1 add-on
            boost = min(max(float(raw_score[i]) / 5.0, 0.0), 0.25)
            level = min(1.0, level + 0.15 * boost)
        scores[i] = round(level * float(config.weight_anomaly), 4)
        reasons[i] = reason
    return pd.DataFrame(
        {
            "event_id": events["event_id"].astype(str).to_numpy(),
            "anomaly_priority_score": scores,
            "anomaly_priority_reason": reasons,
        }
    )
