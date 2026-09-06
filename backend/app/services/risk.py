"""
Backend Phase 11: incremental Stage VI Risk Prioritization.

Loads one ThermalEvent (with I.7 already applied), calls AIML ``realtime.risk``,
and writes Stage VI columns onto **that event only**.

``risk_score`` is decision-support prioritization — not fire probability.
Does not modify I.3–I.7 fields.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.thermal_event import ThermalEvent

logger = logging.getLogger(__name__)

_AIML_ROOT = Path(__file__).resolve().parents[3] / "aiml"
if str(_AIML_ROOT) not in sys.path:
    sys.path.insert(0, str(_AIML_ROOT))

from realtime.risk import RiskPrioritizationResult, process_event_risk  # noqa: E402
from src.risk_prioritization.config import RiskPrioritizationConfig  # noqa: E402
from src.risk_prioritization.risk_schema import RISK_APPEND_COLUMNS  # noqa: E402


@dataclass
class RiskPrioritizationStats:
    events_updated: int = 0
    event_ids: list[str] = field(default_factory=list)
    by_event: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_RISK_INPUT_ATTRS: tuple[str, ...] = (
    "peak_frp",
    "detection_count",
    "observed_duration_hours",
    "persistence_label",
    "anomaly_status",
    "anomaly_score",
    "facility_association_method",
    "facility_attribution_confidence",
    "facility_type",
    "baseline_history_status",
    "industrial_evidence_score",
    "evidence_strength",
    "source_intelligence_candidate",
    "evidence_uncertainty",
    "evidence_sufficiency",
    "limiting_evidence_codes",
    "sta_domain_available",
    "environmental_domain_available",
    "candidate_is_ground_truth",
    "evidence_fusion_score",
    "evidence_coverage",
)


def _event_to_row(event: ThermalEvent) -> dict[str, Any]:
    row: dict[str, Any] = {"event_id": event.event_id}
    for attr in _RISK_INPUT_ATTRS:
        if hasattr(event, attr):
            row[attr] = getattr(event, attr)
    return row


def apply_risk_result(event: ThermalEvent, result: RiskPrioritizationResult) -> None:
    """Write Stage VI columns onto one ThermalEvent (does not touch I.3–I.7)."""
    for col in RISK_APPEND_COLUMNS:
        if hasattr(event, col):
            setattr(event, col, result.values.get(col))


def refresh_event_risk(
    session: Session,
    event_id: str,
    *,
    config: Optional[RiskPrioritizationConfig] = None,
) -> RiskPrioritizationResult:
    """Phase 11 entry: recompute Stage VI risk for one event after Phase 10."""
    event = session.scalar(select(ThermalEvent).where(ThermalEvent.event_id == event_id))
    if event is None:
        raise RuntimeError(f"Event missing for risk refresh: {event_id}")

    events_df = pd.DataFrame([_event_to_row(event)])
    result = process_event_risk(events_df, event_id, config=config)
    apply_risk_result(event, result)
    session.flush()
    logger.info(
        "Phase 11 risk: event=%s score=%s priority=%s context=%s",
        event_id,
        result.risk_score,
        result.investigation_priority,
        result.get("industrial_context"),
    )
    return result
