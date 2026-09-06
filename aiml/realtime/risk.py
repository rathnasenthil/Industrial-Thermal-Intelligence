"""
Incremental Stage VI Decision & Risk Prioritization (AIML realtime).

Thin adapter around batch ``run_risk_prioritization`` for **one** event.

``risk_score`` is a deterministic decision-support prioritization score
(0–100), NOT a probability of industrial fire or fire existence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from src.risk_prioritization.config import RiskPrioritizationConfig
from src.risk_prioritization.risk_pipeline import run_risk_prioritization
from src.risk_prioritization.risk_schema import RISK_APPEND_COLUMNS

REQUIRED_EVENT_COLUMNS: tuple[str, ...] = ("event_id",)


def _coerce_value(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    if isinstance(val, str) and val.strip().lower() == "nan":
        return None
    try:
        import numpy as np

        if isinstance(val, np.generic):
            return val.item()
    except Exception:
        pass
    return val


@dataclass(frozen=True)
class RiskPrioritizationResult:
    """Stage VI fields for one event (batch ``RISK_APPEND_COLUMNS``)."""

    event_id: str
    values: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"event_id": self.event_id, **self.values}

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    @property
    def risk_score(self) -> float | None:
        v = self.values.get("risk_score")
        return float(v) if v is not None else None

    @property
    def investigation_priority(self) -> str | None:
        v = self.values.get("investigation_priority")
        return str(v) if v is not None else None


def _row_to_result(row: pd.Series) -> RiskPrioritizationResult:
    values: dict[str, Any] = {}
    for col in RISK_APPEND_COLUMNS:
        values[col] = _coerce_value(row.get(col))
    return RiskPrioritizationResult(event_id=str(row["event_id"]), values=values)


def process_event_risk(
    events_df: pd.DataFrame,
    event_id: str,
    *,
    config: Optional[RiskPrioritizationConfig] = None,
) -> RiskPrioritizationResult:
    """
    Compute Stage VI risk prioritization for **one** event using batch semantics.

    Args:
        events_df: Must contain the current event with thermal + I.7 fields
            expected by batch Stage VI.
        event_id: Event to score.
        config: Defaults to batch ``RiskPrioritizationConfig()``.
    """
    cfg = config or RiskPrioritizationConfig()
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

    result = run_risk_prioritization(work, cfg)
    for col in RISK_APPEND_COLUMNS:
        if col not in result.events_df.columns:
            raise RuntimeError(f"Batch Stage VI output missing column: {col}")

    row = result.events_df.loc[result.events_df["event_id"].astype(str) == eid].iloc[0]
    out = _row_to_result(row)
    score = out.risk_score
    if score is not None and (score < 0 or score > 100):
        raise RuntimeError(f"risk_score out of range: {score}")
    return out
