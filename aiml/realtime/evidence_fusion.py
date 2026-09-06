"""
Incremental Stage I.7 Evidence Fusion / Source Intelligence (AIML realtime).

Thin adapter around batch ``run_evidence_fusion`` for **one** event.

Evidence fusion is interpretation / decision-support — not ground truth,
industrial-fire classification (unless batch candidate vocab), or risk scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from src.evidence_fusion.config import EvidenceFusionConfig
from src.evidence_fusion.fusion_pipeline import run_evidence_fusion
from src.evidence_fusion.fusion_schema import FUSION_COLUMNS

REQUIRED_EVENT_COLUMNS: tuple[str, ...] = ("event_id",)


def _coerce_value(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    if isinstance(val, str) and val.strip().lower() == "nan":
        return None
    # numpy bool_ / int_
    try:
        import numpy as np

        if isinstance(val, np.generic):
            return val.item()
    except Exception:
        pass
    return val


@dataclass(frozen=True)
class EvidenceFusionResult:
    """I.7 fields for one event (batch ``FUSION_COLUMNS``)."""

    event_id: str
    values: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"event_id": self.event_id, **self.values}

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


def _row_to_result(row: pd.Series) -> EvidenceFusionResult:
    values: dict[str, Any] = {}
    for col in FUSION_COLUMNS:
        values[col] = _coerce_value(row.get(col))
    # Hard semantic: never claim ground truth
    values["candidate_is_ground_truth"] = False
    return EvidenceFusionResult(event_id=str(row["event_id"]), values=values)


def process_event_evidence_fusion(
    events_df: pd.DataFrame,
    event_id: str,
    *,
    config: Optional[EvidenceFusionConfig] = None,
) -> EvidenceFusionResult:
    """
    Compute I.7 evidence fusion for **one** event using batch semantics.

    Args:
        events_df: Must contain the current event with upstream G→I.6 fields
            that batch I.7 reads (persistence, facility, anomaly, STA, env).
        event_id: Event to fuse.
        config: Defaults to batch ``EvidenceFusionConfig()``.
    """
    cfg = config or EvidenceFusionConfig()
    eid = str(event_id)

    if events_df is None or events_df.empty:
        raise ValueError("events_df must contain the current event.")
    for col in REQUIRED_EVENT_COLUMNS:
        if col not in events_df.columns:
            raise ValueError(f"Events dataframe missing required column: {col}")
    if not (events_df["event_id"].astype(str) == eid).any():
        raise ValueError(f"event_id={eid} not present in events_df.")

    work = events_df.loc[events_df["event_id"].astype(str) == eid].copy()
    if len(work) != 1:
        raise ValueError(f"Expected exactly one row for event_id={eid}, got {len(work)}.")

    result = run_evidence_fusion(work, cfg)
    for col in FUSION_COLUMNS:
        if col not in result.events_df.columns:
            raise RuntimeError(f"Batch I.7 output missing column: {col}")

    row = result.events_df.loc[result.events_df["event_id"].astype(str) == eid].iloc[0]
    out = _row_to_result(row)
    if out.values.get("candidate_is_ground_truth") is not False:
        raise RuntimeError("I.7 candidate_is_ground_truth must be False.")
    return out
