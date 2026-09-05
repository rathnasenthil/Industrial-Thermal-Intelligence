"""Thermal severity component for Stage VI.

Uses log1p(FRP), capped detection_count and duration contributions.
Deterministic, distribution-aware ranking within the batch when possible.
NOT a fire probability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk_prioritization.config import (
    SEVERITY_EXTREME,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MODERATE,
    RiskPrioritizationConfig,
)


def compute_thermal_severity(
    events: pd.DataFrame,
    config: RiskPrioritizationConfig,
) -> pd.DataFrame:
    """Return thermal_severity_score (0..weight_thermal) and band per event."""
    n = len(events)
    frp = pd.to_numeric(events.get("peak_frp"), errors="coerce").to_numpy(dtype=float)
    dets = pd.to_numeric(events.get("detection_count"), errors="coerce").to_numpy(dtype=float)
    dur = pd.to_numeric(events.get("observed_duration_hours"), errors="coerce").to_numpy(dtype=float)

    # log1p FRP with cap — handles skew; missing → 0 contribution (documented unavailable)
    frp_safe = np.where(np.isfinite(frp) & (frp > 0), frp, 0.0)
    frp_log = np.log1p(frp_safe)
    frp_norm = np.clip(frp_log / max(config.thermal_frp_log_cap, 1e-9), 0.0, 1.0)

    det_safe = np.where(np.isfinite(dets) & (dets > 0), dets, 0.0)
    det_norm = np.clip(det_safe / max(config.thermal_detection_cap, 1e-9), 0.0, 1.0)

    dur_safe = np.where(np.isfinite(dur) & (dur > 0), dur, 0.0)
    dur_norm = np.clip(dur_safe / max(config.thermal_duration_hours_cap, 1e-9), 0.0, 1.0)

    # Weighted blend within thermal component: FRP 60%, detections 25%, duration 15%
    raw = 0.60 * frp_norm + 0.25 * det_norm + 0.15 * dur_norm

    # Batch percentile boost for relative extremeness (deterministic via average rank)
    # Only among finite FRP > 0
    ranks = np.zeros(n, dtype=float)
    valid = np.isfinite(frp) & (frp > 0)
    if valid.any():
        # average rank percentile
        order = np.argsort(np.argsort(frp_safe))
        pct = order / max(n - 1, 1)
        ranks = np.where(valid, pct, 0.0)

    # Mix absolute normalized intensity with relative rank (70/30)
    combined = 0.70 * raw + 0.30 * ranks
    score = np.round(combined * float(config.weight_thermal), 4)

    bands = np.empty(n, dtype=object)
    for i, s in enumerate(score):
        frac = float(s) / float(config.weight_thermal) if config.weight_thermal else 0.0
        if frac >= 0.85:
            bands[i] = SEVERITY_EXTREME
        elif frac >= 0.60:
            bands[i] = SEVERITY_HIGH
        elif frac >= 0.30:
            bands[i] = SEVERITY_MODERATE
        else:
            bands[i] = SEVERITY_LOW

    return pd.DataFrame(
        {
            "event_id": events["event_id"].astype(str).to_numpy(),
            "thermal_severity_score": score,
            "thermal_severity_band": bands,
        }
    )
