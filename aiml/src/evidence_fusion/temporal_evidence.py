"""Temporal evidence extraction from G.1 persistence + I.4 anomaly fields."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evidence_fusion.fusion_schema import clean_text


def extract_temporal_evidence(events_df: pd.DataFrame) -> pd.DataFrame:
    """Build temporal evidence columns for every event.

    Persistence and anomaly fields are almost always present after G.1/I.4.
    If either core column is missing, temporal_evidence_available=False and
    signals are UNAVAILABLE — never fabricated.
    """
    n = len(events_df)
    event_ids = events_df["event_id"].astype(str).to_numpy()

    has_persistence = "persistence_label" in events_df.columns
    has_anomaly = "anomaly_status" in events_df.columns
    available = has_persistence or has_anomaly

    persistence = (
        events_df["persistence_label"].map(lambda v: clean_text(v, "UNAVAILABLE"))
        if has_persistence
        else pd.Series(["UNAVAILABLE"] * n, index=events_df.index)
    )
    anomaly = (
        events_df["anomaly_status"].map(lambda v: clean_text(v, "UNAVAILABLE"))
        if has_anomaly
        else pd.Series(["UNAVAILABLE"] * n, index=events_df.index)
    )

    summaries: list[str] = []
    for p, a in zip(persistence.tolist(), anomaly.tolist()):
        parts = []
        if has_persistence:
            parts.append(f"persistence={p}")
        if has_anomaly:
            parts.append(f"anomaly={a}")
        if not parts:
            summaries.append("temporal evidence unavailable")
        else:
            summaries.append("; ".join(parts))

    return pd.DataFrame(
        {
            "event_id": event_ids,
            "temporal_evidence_available": np.full(n, bool(available)),
            "temporal_persistence_signal": persistence.to_numpy(),
            "temporal_anomaly_signal": anomaly.to_numpy(),
            "temporal_evidence_summary": np.asarray(summaries, dtype=object),
        }
    )
