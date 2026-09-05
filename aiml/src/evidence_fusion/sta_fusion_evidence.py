"""NASA STA evidence summarization for Stage I.7 fusion.

Distinguishes:
- domain unavailable (I.5 columns absent / STA never integrated)
- domain available with NO_STA_ASSOCIATION (looked, no match)
Missing STA is NEVER treated as anti-industrial evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evidence_fusion.fusion_schema import STA_INPUT_COLUMNS, clean_text


def extract_sta_evidence(events_df: pd.DataFrame) -> pd.DataFrame:
    """Build STA domain evidence columns for every event."""
    n = len(events_df)
    event_ids = events_df["event_id"].astype(str).to_numpy()
    domain_available = all(col in events_df.columns for col in ("sta_association_status",))

    if not domain_available:
        return pd.DataFrame(
            {
                "event_id": event_ids,
                "sta_domain_available": np.full(n, False),
                "sta_association_signal": np.full(n, "UNAVAILABLE", dtype=object),
                "sta_layer_signal": np.full(n, None, dtype=object),
                "sta_quality_signal": np.full(n, "UNAVAILABLE", dtype=object),
                "sta_evidence_summary": np.full(
                    n, "STA domain unavailable (I.5 columns absent)", dtype=object
                ),
            }
        )

    status = events_df["sta_association_status"].map(
        lambda v: clean_text(v, "NO_STA_ASSOCIATION")
    )
    quality = (
        events_df["sta_evidence_quality"].map(lambda v: clean_text(v, "NONE"))
        if "sta_evidence_quality" in events_df.columns
        else pd.Series(["NONE"] * n, index=events_df.index)
    )
    layer = (
        events_df["sta_layer_type"].map(lambda v: clean_text(v, None))
        if "sta_layer_type" in events_df.columns
        else pd.Series([None] * n, index=events_df.index)
    )

    # Per-row: if sta_evidence_available column exists and is False with
    # NO_STA_ASSOCIATION, domain is still "available" (processed) — the match
    # simply did not occur. Only missing columns mean UNAVAILABLE domain.
    summaries: list[str] = []
    for st, q, ly in zip(status.tolist(), quality.tolist(), layer.tolist()):
        parts = [f"sta_status={st}", f"quality={q}"]
        if ly is not None:
            parts.append(f"layer={ly}")
        if st == "NO_STA_ASSOCIATION":
            parts.append("no_match_is_not_anti_industrial")
        summaries.append("; ".join(parts))

    return pd.DataFrame(
        {
            "event_id": event_ids,
            "sta_domain_available": np.full(n, True),
            "sta_association_signal": status.to_numpy(),
            "sta_layer_signal": layer.to_numpy(),
            "sta_quality_signal": quality.to_numpy(),
            "sta_evidence_summary": np.asarray(summaries, dtype=object),
        }
    )
