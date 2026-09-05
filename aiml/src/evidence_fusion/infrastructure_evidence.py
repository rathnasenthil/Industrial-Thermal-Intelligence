"""Infrastructure evidence extraction from I.2 association + I.3/I.4 history."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evidence_fusion.config import CONFIRMED_FACILITY_METHODS
from src.evidence_fusion.fusion_schema import clean_text


def _association_signal(method: str | None) -> str:
    if method is None or method == "UNAVAILABLE":
        return "UNAVAILABLE"
    if method == "NO_FACILITY_ASSOCIATION":
        return "NONE"
    if method == "AMBIGUOUS":
        return "AMBIGUOUS"
    if method in CONFIRMED_FACILITY_METHODS:
        return "CONFIRMED"
    return "OTHER"


def extract_infrastructure_evidence(events_df: pd.DataFrame) -> pd.DataFrame:
    """Build infrastructure evidence columns for every event."""
    n = len(events_df)
    event_ids = events_df["event_id"].astype(str).to_numpy()
    has_method = "facility_association_method" in events_df.columns
    available = has_method

    methods = (
        events_df["facility_association_method"].map(lambda v: clean_text(v, "UNAVAILABLE"))
        if has_method
        else pd.Series(["UNAVAILABLE"] * n, index=events_df.index)
    )
    conf = (
        events_df["facility_attribution_confidence"].map(lambda v: clean_text(v, "NONE"))
        if "facility_attribution_confidence" in events_df.columns
        else pd.Series(["UNAVAILABLE"] * n, index=events_df.index)
    )
    ftype = (
        events_df["facility_type"].map(lambda v: clean_text(v, None))
        if "facility_type" in events_df.columns
        else pd.Series([None] * n, index=events_df.index)
    )
    history = (
        events_df["baseline_history_status"].map(lambda v: clean_text(v, "NOT_APPLICABLE"))
        if "baseline_history_status" in events_df.columns
        else pd.Series(["UNAVAILABLE"] * n, index=events_df.index)
    )

    assoc_signals = [_association_signal(m) for m in methods.tolist()]
    type_signals: list[str | None] = []
    for signal, ft in zip(assoc_signals, ftype.tolist()):
        if signal in ("NONE", "UNAVAILABLE", "AMBIGUOUS"):
            type_signals.append(None if signal != "AMBIGUOUS" else "AMBIGUOUS")
        else:
            type_signals.append(ft)

    summaries: list[str] = []
    for method, signal, c, ft, hist in zip(
        methods.tolist(), assoc_signals, conf.tolist(), type_signals, history.tolist()
    ):
        if not available:
            summaries.append("infrastructure evidence unavailable")
            continue
        parts = [f"association={method}", f"signal={signal}", f"confidence={c}"]
        if ft is not None:
            parts.append(f"facility_type={ft}")
        parts.append(f"history={hist}")
        summaries.append("; ".join(parts))

    return pd.DataFrame(
        {
            "event_id": event_ids,
            "infrastructure_evidence_available": np.full(n, bool(available)),
            "infrastructure_association_signal": np.asarray(assoc_signals, dtype=object),
            "infrastructure_facility_type_signal": np.asarray(type_signals, dtype=object),
            "infrastructure_confidence_signal": conf.to_numpy(),
            "infrastructure_history_signal": history.to_numpy(),
            "infrastructure_evidence_summary": np.asarray(summaries, dtype=object),
        }
    )
