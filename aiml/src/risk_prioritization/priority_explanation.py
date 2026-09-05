"""Explainability fields for Stage VI priority decisions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk_prioritization.config import (
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    SEVERITY_EXTREME,
    SEVERITY_HIGH,
    RiskPrioritizationConfig,
)


def build_priority_explanations(
    scored: pd.DataFrame,
    config: RiskPrioritizationConfig,
) -> pd.DataFrame:
    """Generate per-event reasons/warnings/dominant factors from actual scores."""
    n = len(scored)
    reasons = np.empty(n, dtype=object)
    warnings = np.empty(n, dtype=object)
    dominant = np.empty(n, dtype=object)

    for i in range(n):
        row = scored.iloc[i]
        r_list: list[str] = []
        w_list: list[str] = []
        d_list: list[tuple[float, str]] = []

        thermal = float(row["thermal_severity_score"])
        persist = float(row["persistence_priority_score"])
        anomaly = float(row["anomaly_priority_score"])
        facility = float(row["facility_context_score"])
        ie = float(row["industrial_evidence_component"])

        band = str(row["thermal_severity_band"])
        if band in (SEVERITY_HIGH, SEVERITY_EXTREME):
            r_list.append(f"HIGH_THERMAL_SEVERITY:{band}")
        d_list.append((thermal, "THERMAL_SEVERITY"))

        preason = str(row["persistence_priority_reason"])
        if persist >= 0.5 * config.weight_persistence:
            r_list.append(preason)
        d_list.append((persist, "PERSISTENCE"))

        areason = str(row["anomaly_priority_reason"])
        if anomaly >= 0.4 * config.weight_anomaly:
            r_list.append(areason)
        d_list.append((anomaly, "ANOMALY"))

        freason = str(row["facility_context_reason"])
        if facility >= 0.4 * config.weight_facility:
            r_list.append("HIGH_FACILITY_CONTEXT")
            if "TYPE_" in freason:
                r_list.append(freason.split(";")[0] if ";" in freason else freason)
        d_list.append((facility, "FACILITY_CONTEXT"))

        if ie >= 0.4 * config.weight_industrial_evidence:
            r_list.append("ELEVATED_INDUSTRIAL_EVIDENCE")
        d_list.append((ie, "INDUSTRIAL_EVIDENCE"))

        if str(row["investigation_priority"]) in (PRIORITY_HIGH, PRIORITY_CRITICAL):
            r_list.append(f"PRIORITY_{row['investigation_priority']}")

        # Warnings from limiting / uncertainty (unavailable ≠ negative)
        lim = str(row.get("risk_limiting_evidence_codes") or "")
        for code in lim.split(";"):
            code = code.strip()
            if code in {
                "STA_UNAVAILABLE",
                "ENVIRONMENTAL_CONTEXT_UNAVAILABLE",
                "NO_FACILITY_ASSOCIATION",
                "AMBIGUOUS_FACILITY_ASSOCIATION",
                "INSUFFICIENT_HISTORY",
            }:
                w_list.append(code)

        # Ensure uniqueness preserving order
        def _uniq(items: list[str]) -> str:
            out: list[str] = []
            for x in items:
                if x and x not in out:
                    out.append(x)
            return ";".join(out)

        d_list.sort(key=lambda t: (-t[0], t[1]))
        dominant[i] = ";".join([name for score, name in d_list[:3] if score > 0] or ["NONE"])
        reasons[i] = _uniq(r_list) or "NO_ELEVATED_PRIORITY_DRIVERS"
        warnings[i] = _uniq(w_list)

    return pd.DataFrame(
        {
            "event_id": scored["event_id"].astype(str).to_numpy(),
            "priority_reasons": reasons,
            "priority_warnings": warnings,
            "dominant_risk_factors": dominant,
        }
    )
