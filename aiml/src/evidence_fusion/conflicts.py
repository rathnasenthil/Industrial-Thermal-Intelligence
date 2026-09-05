"""Explicit conflict detection across available evidence domains.

Conflicts are only declared when *available* evidence points in
genuinely different directions. Missing/null evidence never creates a
conflict and never acts as a negative score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_evidence_conflicts(
    infrastructure: pd.DataFrame,
    sta: pd.DataFrame,
    environmental: pd.DataFrame,
) -> pd.DataFrame:
    """Return conflict flags/codes/summaries aligned to event_id order."""
    event_ids = infrastructure["event_id"].astype(str).to_numpy()
    n = len(event_ids)

    infra_signal = infrastructure["infrastructure_association_signal"].to_numpy()
    sta_signal = sta["sta_association_signal"].to_numpy()
    sta_available = sta["sta_domain_available"].astype(bool).to_numpy()
    env_available = environmental["environmental_domain_available"].astype(bool).to_numpy()
    veg = environmental["environmental_vegetation_signal"].to_numpy()
    ag = environmental["environmental_agriculture_signal"].to_numpy()
    built = environmental["environmental_builtup_signal"].to_numpy()

    flags: list[bool] = []
    codes: list[str] = []
    summaries: list[str] = []

    for i in range(n):
        found: list[str] = []
        confirmed = infra_signal[i] == "CONFIRMED"
        ambiguous_fac = infra_signal[i] == "AMBIGUOUS"
        none_fac = infra_signal[i] == "NONE"

        # Facility vs agriculture / vegetation (only when env present).
        if confirmed and env_available[i]:
            if ag[i] == "PRESENT":
                found.append("FACILITY_VS_AGRICULTURE")
            if veg[i] == "PRESENT" and built[i] != "PRESENT":
                found.append("FACILITY_VS_VEGETATION")

        # STA associated without facility, with vegetation present.
        if (
            sta_available[i]
            and sta_signal[i] in ("STA_ASSOCIATED", "AMBIGUOUS")
            and none_fac
            and env_available[i]
            and veg[i] == "PRESENT"
        ):
            found.append("STA_WITHOUT_FACILITY_WITH_VEGETATION")

        # Ambiguous facility + STA associated (competing industrial-ish cues).
        if ambiguous_fac and sta_available[i] and sta_signal[i] == "STA_ASSOCIATED":
            found.append("AMBIGUOUS_FACILITY_WITH_STA")

        # Agriculture + vegetation both PRESENT with no facility (competing env).
        if none_fac and env_available[i] and ag[i] == "PRESENT" and veg[i] == "PRESENT":
            found.append("AGRICULTURE_AND_VEGETATION_BOTH_PRESENT")

        # Deduplicate while preserving order.
        ordered: list[str] = []
        for code in found:
            if code not in ordered:
                ordered.append(code)

        flags.append(bool(ordered))
        codes.append(";".join(ordered) if ordered else "")
        if ordered:
            summaries.append("conflicting available evidence: " + ", ".join(ordered))
        else:
            summaries.append("no explicit conflict among available evidence")

    return pd.DataFrame(
        {
            "event_id": event_ids,
            "evidence_conflict_flag": np.asarray(flags, dtype=bool),
            "evidence_conflict_codes": np.asarray(codes, dtype=object),
            "evidence_conflict_summary": np.asarray(summaries, dtype=object),
        }
    )
