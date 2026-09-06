"""
Backend Phase 10: incremental Stage I.7 Evidence Fusion.

Loads one ThermalEvent upstream context, calls AIML ``realtime.evidence_fusion``,
and writes I.7 columns onto **that event only**.

Does not modify I.3–I.6 fields. Does not compute risk. Does not claim ground truth.
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

from realtime.evidence_fusion import EvidenceFusionResult, process_event_evidence_fusion  # noqa: E402
from src.evidence_fusion.config import EvidenceFusionConfig  # noqa: E402
from src.evidence_fusion.fusion_schema import FUSION_COLUMNS  # noqa: E402


@dataclass
class EvidenceFusionStats:
    events_updated: int = 0
    event_ids: list[str] = field(default_factory=list)
    by_event: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Columns batch I.7 / upstream extractors commonly read from the events frame.
_FUSION_INPUT_ATTRS: tuple[str, ...] = (
    "persistence_label",
    "anomaly_status",
    "anomaly_score",
    "anomaly_confidence",
    "peak_frp_deviation",
    "event_size_deviation",
    "duration_deviation",
    "distance_deviation",
    "persistence_deviation",
    "monthly_deviation",
    "baseline_history_status",
    "facility_association_method",
    "facility_attribution_confidence",
    "facility_type",
    "facility_id",
    "facility_distance_km",
    "candidate_facility_count",
    "sta_association_status",
    "sta_evidence_quality",
    "sta_layer_type",
    "sta_evidence_available",
    "sta_match_count",
    "primary_sta_id",
    "sta_temporal_relation",
    "landcover_available",
    "vegetation_context_available",
    "builtup_context_available",
    "water_context_available",
    "agriculture_context_available",
    "satellite_context_available",
    "vegetation_present",
    "agriculture_present",
    "builtup_present",
    "water_present",
    "dominant_landcover_class",
)


def _event_to_row(event: ThermalEvent) -> dict[str, Any]:
    row: dict[str, Any] = {"event_id": event.event_id}
    for attr in _FUSION_INPUT_ATTRS:
        if hasattr(event, attr):
            row[attr] = getattr(event, attr)
    return row


def apply_evidence_fusion_result(
    event: ThermalEvent, result: EvidenceFusionResult
) -> None:
    """Write I.7 columns onto one ThermalEvent (does not touch I.3–I.6 / risk)."""
    for col in FUSION_COLUMNS:
        if hasattr(event, col):
            setattr(event, col, result.values.get(col))


def refresh_event_evidence_fusion(
    session: Session,
    event_id: str,
    *,
    config: Optional[EvidenceFusionConfig] = None,
) -> EvidenceFusionResult:
    """Phase 10 entry: recompute I.7 for one event after Phase 9."""
    event = session.scalar(select(ThermalEvent).where(ThermalEvent.event_id == event_id))
    if event is None:
        raise RuntimeError(f"Event missing for evidence fusion refresh: {event_id}")

    events_df = pd.DataFrame([_event_to_row(event)])
    result = process_event_evidence_fusion(events_df, event_id, config=config)
    apply_evidence_fusion_result(event, result)
    session.flush()
    logger.info(
        "Phase 10 fusion: event=%s candidate=%s strength=%s fusion_score=%s",
        event_id,
        result.get("source_intelligence_candidate"),
        result.get("evidence_strength"),
        result.get("evidence_fusion_score"),
    )
    return result
