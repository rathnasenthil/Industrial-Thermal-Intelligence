"""
Backend Phase 6: incremental Stage I.3 facility thermal fingerprinting.

Loads confirmed (+ ambiguous-candidate) events for affected facilities,
calls AIML ``realtime.facility_fingerprint``, and upserts:

- ``facility_thermal_fingerprints``
- ``facility_monthly_thermal_profile``

Descriptive baseline only — not anomaly detection / fire classification.
Does **not** call ``run_facility_fingerprinting()`` over all facilities.
Does **not** overwrite ThermalEvent baseline/anomaly fields.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models.event_facility_candidate import EventFacilityCandidate
from app.models.facility import Facility
from app.models.facility_monthly_thermal_profile import FacilityMonthlyThermalProfile
from app.models.facility_thermal_fingerprint import FacilityThermalFingerprint
from app.models.thermal_event import ThermalEvent

logger = logging.getLogger(__name__)

_AIML_ROOT = Path(__file__).resolve().parents[3] / "aiml"
if str(_AIML_ROOT) not in sys.path:
    sys.path.insert(0, str(_AIML_ROOT))

from realtime.facility_fingerprint import (  # noqa: E402
    CONFIRMED_ASSOCIATION_METHODS,
    process_facility_fingerprint,
)
from src.fingerprinting.facility_fingerprint import REQUIRED_EVENT_COLUMNS  # noqa: E402
from src.fingerprinting.fingerprint_config import DEFAULT_CONFIG, FingerprintConfig  # noqa: E402
from src.infrastructure.facility_association import (  # noqa: E402
    AMBIGUOUS,
    NO_FACILITY_ASSOCIATION,
)

_FINGERPRINT_COUNT_DEFAULTS = {
    "event_count": 0,
    "detection_count": 0,
    "observation_day_count": 0,
    "active_month_count": 0,
    "day_event_count": 0,
    "night_event_count": 0,
    "persistent_event_count": 0,
    "recurring_event_count": 0,
    "short_lived_event_count": 0,
    "insufficient_observations_event_count": 0,
    "within_facility_count": 0,
    "intersects_facility_count": 0,
    "near_facility_count": 0,
    "high_confidence_event_count": 0,
    "medium_confidence_event_count": 0,
    "low_confidence_event_count": 0,
    "ambiguous_candidate_opportunity_count": 0,
    "fingerprint_observation_count": 0,
}


@dataclass
class FacilityFingerprintStats:
    facilities_refreshed: int = 0
    facility_ids: list[str] = field(default_factory=list)
    by_facility: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _candidate_facility_ids_csv(session: Session, event_id: str) -> str:
    """Reconstruct batch ``candidate_facility_ids`` from EventFacilityCandidate."""
    ids = list(
        session.scalars(
            select(EventFacilityCandidate.facility_id)
            .where(EventFacilityCandidate.event_id == event_id)
            .order_by(EventFacilityCandidate.facility_id.asc())
        ).all()
    )
    return ",".join(sorted(str(i) for i in ids if i))


def _event_to_row(session: Session, event: ThermalEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "detection_count": event.detection_count,
        "event_start": event.event_start,
        "event_end": event.event_end,
        "observed_duration_hours": event.observed_duration_hours,
        "distinct_detection_days": event.distinct_detection_days,
        "peak_frp": event.peak_frp,
        "day_detection_count": event.day_detection_count,
        "night_detection_count": event.night_detection_count,
        "persistence_label": event.persistence_label,
        "facility_id": event.facility_id,
        "facility_association_method": event.facility_association_method,
        "facility_attribution_confidence": event.facility_attribution_confidence,
        "facility_distance_km": event.facility_distance_km,
        "candidate_facility_ids": _candidate_facility_ids_csv(session, event.event_id),
    }


def load_events_dataframe_for_facility(session: Session, facility_id: str) -> pd.DataFrame:
    """
    Confirmed events for ``facility_id`` plus AMBIGUOUS events listing it.

    Does not add ``candidate_facility_ids`` to ThermalEvent — reconstructed
    from ``event_facility_candidates`` only.
    """
    confirmed = list(
        session.scalars(
            select(ThermalEvent).where(
                ThermalEvent.facility_id == facility_id,
                ThermalEvent.facility_association_method.in_(
                    list(CONFIRMED_ASSOCIATION_METHODS)
                ),
            )
        ).all()
    )
    ambiguous_event_ids = list(
        session.scalars(
            select(EventFacilityCandidate.event_id)
            .join(ThermalEvent, ThermalEvent.event_id == EventFacilityCandidate.event_id)
            .where(
                EventFacilityCandidate.facility_id == facility_id,
                ThermalEvent.facility_association_method == AMBIGUOUS,
            )
            .distinct()
        ).all()
    )
    ambiguous = []
    if ambiguous_event_ids:
        ambiguous = list(
            session.scalars(
                select(ThermalEvent).where(ThermalEvent.event_id.in_(ambiguous_event_ids))
            ).all()
        )

    rows = [_event_to_row(session, ev) for ev in confirmed + ambiguous]
    if not rows:
        return pd.DataFrame(columns=list(REQUIRED_EVENT_COLUMNS))
    return pd.DataFrame(rows)


def affected_facility_ids_for_event(
    session: Session,
    event_id: str,
    *,
    previous_facility_id: Optional[str] = None,
) -> list[str]:
    """
    Facilities whose I.3 rows must be refreshed after Phase 5.

    - Confirmed: selected facility (+ previous if association moved).
    - AMBIGUOUS: all candidate facilities (opportunity count only).
    - NO_FACILITY_ASSOCIATION: none (unless previous confirmed facility lost the event).
    """
    event = session.scalar(select(ThermalEvent).where(ThermalEvent.event_id == event_id))
    if event is None:
        return []

    affected: set[str] = set()
    method = event.facility_association_method

    if previous_facility_id:
        affected.add(previous_facility_id)

    if method in CONFIRMED_ASSOCIATION_METHODS and event.facility_id:
        affected.add(event.facility_id)
    elif method == AMBIGUOUS:
        for fid in session.scalars(
            select(EventFacilityCandidate.facility_id).where(
                EventFacilityCandidate.event_id == event_id
            )
        ):
            if fid:
                affected.add(fid)
    elif method == NO_FACILITY_ASSOCIATION:
        pass

    return sorted(affected)


def _apply_fingerprint_row(
    row: FacilityThermalFingerprint, data: dict[str, Any]
) -> None:
    for key, default in _FINGERPRINT_COUNT_DEFAULTS.items():
        setattr(row, key, int(data.get(key) if data.get(key) is not None else default))

    row.facility_name = data.get("facility_name")
    row.facility_type = data.get("facility_type")
    row.first_observation_date = data.get("first_observation_date")
    row.last_observation_date = data.get("last_observation_date")
    row.observation_span_days = data.get("observation_span_days")
    row.day_event_fraction = data.get("day_event_fraction")
    row.night_event_fraction = data.get("night_event_fraction")

    for prefix in ("peak_frp", "event_size", "duration_hours", "distance_km"):
        for suffix in ("median", "mad", "p25", "p75", "p90", "max"):
            col = f"{prefix}_{suffix}"
            setattr(row, col, data.get(col))

    for frac in (
        "persistent_event_fraction",
        "recurring_event_fraction",
        "short_lived_event_fraction",
    ):
        setattr(row, frac, data.get(frac))

    row.fingerprint_status = str(data.get("fingerprint_status") or "NO_OBSERVATIONS")
    row.updated_at = datetime.now(timezone.utc)


def upsert_facility_fingerprint(
    session: Session,
    fingerprint: dict[str, Any],
    monthly_rows: Sequence[dict[str, Any]],
) -> None:
    facility_id = str(fingerprint["facility_id"])
    row = session.scalar(
        select(FacilityThermalFingerprint).where(
            FacilityThermalFingerprint.facility_id == facility_id
        )
    )
    if row is None:
        row = FacilityThermalFingerprint(facility_id=facility_id)
        session.add(row)
    _apply_fingerprint_row(row, fingerprint)

    session.execute(
        delete(FacilityMonthlyThermalProfile).where(
            FacilityMonthlyThermalProfile.facility_id == facility_id
        )
    )
    for m in monthly_rows:
        session.add(
            FacilityMonthlyThermalProfile(
                facility_id=facility_id,
                month=int(m["month"]),
                event_count=int(m["event_count"]),
                detection_count=int(m["detection_count"]),
                event_fraction=m.get("event_fraction"),
            )
        )
    session.flush()


def refresh_facility_fingerprint(
    session: Session,
    facility_id: str,
    *,
    config: Optional[FingerprintConfig] = None,
) -> dict[str, Any]:
    """Recompute and persist I.3 for one facility only."""
    cfg = config or DEFAULT_CONFIG
    facility = session.scalar(select(Facility).where(Facility.facility_id == facility_id))
    if facility is None:
        raise RuntimeError(f"Facility missing for fingerprint refresh: {facility_id}")

    events_df = load_events_dataframe_for_facility(session, facility_id)
    result = process_facility_fingerprint(
        {
            "facility_id": facility.facility_id,
            "facility_name": facility.facility_name,
            "facility_type": facility.facility_type,
        },
        events_df,
        config=cfg,
    )
    upsert_facility_fingerprint(session, result.fingerprint, result.monthly_profile)
    return result.to_dict()


def refresh_fingerprints_for_event(
    session: Session,
    event_id: str,
    *,
    previous_facility_id: Optional[str] = None,
    config: Optional[FingerprintConfig] = None,
) -> FacilityFingerprintStats:
    """
    Phase 6 entry: refresh only facilities affected by this event's I.2 result.
    """
    cfg = config or DEFAULT_CONFIG
    stats = FacilityFingerprintStats()
    for fid in affected_facility_ids_for_event(
        session, event_id, previous_facility_id=previous_facility_id
    ):
        payload = refresh_facility_fingerprint(session, fid, config=cfg)
        stats.facilities_refreshed += 1
        stats.facility_ids.append(fid)
        fp = payload["fingerprint"]
        stats.by_facility[fid] = {
            "event_count": fp.get("event_count"),
            "fingerprint_status": fp.get("fingerprint_status"),
            "ambiguous_candidate_opportunity_count": fp.get(
                "ambiguous_candidate_opportunity_count"
            ),
            "monthly_rows": len(payload["monthly_profile"]),
        }
    if stats.facility_ids:
        logger.info(
            "Phase 6 facility fingerprint: refreshed=%s facilities=%s",
            stats.facilities_refreshed,
            stats.facility_ids,
        )
    return stats
