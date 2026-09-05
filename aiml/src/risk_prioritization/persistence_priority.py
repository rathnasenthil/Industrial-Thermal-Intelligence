"""Persistence / recurrence priority component (G.1).

PERSISTENT != industrial fire. PERSISTENT != dangerous.
Contribution only to investigation priority.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk_prioritization.config import RiskPrioritizationConfig

# Ordinal support within persistence component (0-1), then scaled by weight.
_PERSISTENCE_LEVEL: dict[str, tuple[float, str]] = {
    "RECURRING": (1.0, "RECURRING_ACTIVITY"),
    "PERSISTENT": (0.85, "PERSISTENT_ACTIVITY"),
    "SHORT_LIVED": (0.25, "SHORT_LIVED_ACTIVITY"),
    "INSUFFICIENT_OBSERVATIONS": (0.0, "INSUFFICIENT_OBSERVATIONS"),
}


def compute_persistence_priority(
    events: pd.DataFrame,
    config: RiskPrioritizationConfig,
) -> pd.DataFrame:
    n = len(events)
    labels = (
        events["persistence_label"].astype(str).to_numpy()
        if "persistence_label" in events.columns
        else np.full(n, "INSUFFICIENT_OBSERVATIONS", dtype=object)
    )
    scores = np.zeros(n, dtype=float)
    reasons = np.empty(n, dtype=object)
    for i, lab in enumerate(labels):
        level, reason = _PERSISTENCE_LEVEL.get(lab, (0.0, "PERSISTENCE_UNKNOWN"))
        scores[i] = round(level * float(config.weight_persistence), 4)
        reasons[i] = reason
    return pd.DataFrame(
        {
            "event_id": events["event_id"].astype(str).to_numpy(),
            "persistence_priority_score": scores,
            "persistence_priority_reason": reasons,
        }
    )
