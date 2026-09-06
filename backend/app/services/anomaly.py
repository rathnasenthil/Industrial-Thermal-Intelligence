"""
Backend Phase 7: incremental Stage I.4 temporal anomaly detection.

Loads confirmed events for the current event's facility, calls AIML
``realtime.anomaly``, and writes I.4 columns onto **that event only**.

Walk-forward prior-only scoring — not I.3 fingerprinting, not risk,
not industrial-fire classification.

Does **not** call ``run_anomaly_detection()`` over all events.
Does **not** read ``facility_thermal_fingerprints`` /
``facility_monthly_thermal_profile`` for scoring.
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

from realtime.anomaly import (  # noqa: E402
    AnomalyResult,
    process_event_anomaly,
    unavailable_anomaly_result,
)
from src.anomaly_detection.config import (  # noqa: E402
    DEFAULT_CONFIG,
    REASON_AMBIGUOUS,
    REASON_NO_FACILITY,
    AnomalyConfig,
    CONFIRMED_ASSOCIATION_METHODS,
)
from src.infrastructure.facility_association import (  # noqa: E402
    AMBIGUOUS,
    NO_FACILITY_ASSOCIATION,
)


@dataclass
class AnomalyStats:
    events_scored: int = 0
    event_ids: list[str] = field(default_factory=list)
    by_event: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _event_to_row(event: ThermalEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "event_start": event.event_start,
        "event_end": event.event_end,
        "peak_frp": event.peak_frp,
        "detection_count": event.detection_count,
        "observed_duration_hours": event.observed_duration_hours,
        "persistence_label": event.persistence_label,
        "facility_id": event.facility_id,
        "facility_association_method": event.facility_association_method,
        "facility_distance_km": event.facility_distance_km,
    }


def load_confirmed_events_for_facility(session: Session, facility_id: str) -> pd.DataFrame:
    """All confirmed I.2 associations for one facility (includes current)."""
    events = list(
        session.scalars(
            select(ThermalEvent).where(
                ThermalEvent.facility_id == facility_id,
                ThermalEvent.facility_association_method.in_(
                    list(CONFIRMED_ASSOCIATION_METHODS)
                ),
            )
        ).all()
    )
    if not events:
        return pd.DataFrame(
            columns=[
                "event_id",
                "event_start",
                "event_end",
                "peak_frp",
                "detection_count",
                "observed_duration_hours",
                "persistence_label",
                "facility_id",
                "facility_association_method",
                "facility_distance_km",
            ]
        )
    rows = [_event_to_row(ev) for ev in events]
    return pd.DataFrame(rows)


def apply_anomaly_result(event: ThermalEvent, result: AnomalyResult) -> None:
    """Write I.4 fields onto one ThermalEvent (existing schema only)."""
    event.baseline_observation_count = result.baseline_observation_count
    event.baseline_history_status = result.baseline_history_status
    event.anomaly_unavailable_reason = result.anomaly_unavailable_reason
    event.anomaly_score = result.anomaly_score
    event.anomaly_status = result.anomaly_status
    event.anomaly_confidence = result.anomaly_confidence
    event.peak_frp_deviation = result.peak_frp_deviation
    event.event_size_deviation = result.event_size_deviation
    event.duration_deviation = result.duration_deviation
    event.distance_deviation = result.distance_deviation
    event.persistence_deviation = result.persistence_deviation
    event.monthly_deviation = result.monthly_deviation
    event.features_available = result.features_available
    event.features_evaluated = result.features_evaluated
    event.anomaly_explanation = result.anomaly_explanation


def refresh_event_anomaly(
    session: Session,
    event_id: str,
    *,
    config: Optional[AnomalyConfig] = None,
) -> AnomalyResult:
    """
    Phase 7 entry: recompute I.4 for one event after Phase 5/6.

    Confirmed associations: walk-forward over that facility's confirmed
    history (current included in the frame; excluded from its own baseline
    by batch walk-forward). AMBIGUOUS / NO_FACILITY: unavailable payload.

    Known realtime limitation: does not rescore chronologically later
    historical events at facilities A/B after an association move — only
    the current event is updated.
    """
    cfg = config or DEFAULT_CONFIG
    event = session.scalar(select(ThermalEvent).where(ThermalEvent.event_id == event_id))
    if event is None:
        raise RuntimeError(f"Event missing for anomaly refresh: {event_id}")

    method = event.facility_association_method
    if method == AMBIGUOUS:
        result = unavailable_anomaly_result(event_id, reason=REASON_AMBIGUOUS)
        apply_anomaly_result(event, result)
        session.flush()
        return result

    if (
        method == NO_FACILITY_ASSOCIATION
        or event.facility_id is None
        or method not in CONFIRMED_ASSOCIATION_METHODS
    ):
        result = unavailable_anomaly_result(event_id, reason=REASON_NO_FACILITY)
        apply_anomaly_result(event, result)
        session.flush()
        return result

    events_df = load_confirmed_events_for_facility(session, event.facility_id)
    if events_df.empty or not (events_df["event_id"].astype(str) == event_id).any():
        # Should not happen when method is confirmed; treat as unavailable.
        result = unavailable_anomaly_result(event_id, reason=REASON_NO_FACILITY)
        apply_anomaly_result(event, result)
        session.flush()
        return result

    result = process_event_anomaly(events_df, event_id, config=cfg)
    apply_anomaly_result(event, result)
    session.flush()
    logger.info(
        "Phase 7 anomaly: event=%s status=%s score=%s priors=%s",
        event_id,
        result.anomaly_status,
        result.anomaly_score,
        result.baseline_observation_count,
    )
    return result
