"""
Backend adapter: persist incremental thermal-event formation (Phase 3).

Translates DB rows ↔ AIML ``realtime`` plain objects, allocates stable
``EVT_#######`` IDs from a PostgreSQL sequence, and commits observation
linkage transactionally.

Does not run ST-DBSCAN batch clustering, facility association, anomaly,
fusion, or risk scoring.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from geoalchemy2.elements import WKTElement
from sqlalchemy import func, select, text, update
from sqlalchemy.orm import Session

from app.models.event_detection import EventDetection
from app.models.firms_observation import FirmsObservation
from app.models.thermal_event import ThermalEvent

logger = logging.getLogger(__name__)

# aiml/ on sys.path so ``import realtime`` and ``import src`` both work.
_AIML_ROOT = Path(__file__).resolve().parents[3] / "aiml"
if str(_AIML_ROOT) not in sys.path:
    sys.path.insert(0, str(_AIML_ROOT))

from realtime.config import RealtimeEventConfig, default_realtime_config  # noqa: E402
from realtime.event_matcher import events_to_deactivate  # noqa: E402
from realtime.incremental_processor import process_observation  # noqa: E402
from realtime.schemas import (  # noqa: E402
    ActiveEventState,
    MatchAction,
    ObservationRecord,
)


@dataclass
class EventFormationStats:
    processed: int = 0
    created: int = 0
    matched: int = 0
    skipped_already_assigned: int = 0
    skipped_invalid: int = 0
    deactivated: int = 0
    event_ids_touched: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def allocate_event_id(session: Session) -> str:
    """
    Allocate next stable EVT_####### from PostgreSQL sequence.

    Sequence is advanced atomically (``nextval``) to avoid races.
    Never renumbers existing historical IDs.
    """
    next_num = session.execute(
        text("SELECT nextval('thermal_event_external_id_seq')")
    ).scalar()
    if next_num is None:
        raise RuntimeError("thermal_event_external_id_seq returned NULL")
    return f"EVT_{int(next_num):07d}"


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def observation_to_record(obs: FirmsObservation) -> ObservationRecord:
    if obs.acq_datetime is None:
        raise ValueError(
            f"observation {obs.observation_hash} missing acq_datetime"
        )
    return ObservationRecord(
        observation_hash=obs.observation_hash,
        latitude=float(obs.latitude),
        longitude=float(obs.longitude),
        acq_datetime=_ensure_utc(obs.acq_datetime),
        frp=float(obs.frp) if obs.frp is not None else None,
        bright_ti4=float(obs.bright_ti4) if obs.bright_ti4 is not None else None,
        bright_ti5=float(obs.bright_ti5) if obs.bright_ti5 is not None else None,
        daynight=obs.daynight,
        confidence=obs.confidence,
        event_id=obs.event_id,
    )


def thermal_event_to_state(event: ThermalEvent) -> ActiveEventState:
    last = event.last_detection_at or event.event_end or event.event_start
    if last is None:
        raise ValueError(f"event {event.event_id} has no temporal anchor")
    return ActiveEventState(
        event_id=event.event_id,
        centroid_latitude=float(event.centroid_latitude or 0.0),
        centroid_longitude=float(event.centroid_longitude or 0.0),
        last_detection_at=_ensure_utc(last),
        event_start=_ensure_utc(event.event_start) if event.event_start else None,
        event_end=_ensure_utc(event.event_end) if event.event_end else None,
        detection_count=int(event.detection_count or 0),
        peak_frp=event.peak_frp,
        mean_frp=event.mean_frp,
        total_frp=event.total_frp,
        frp_valid_count=int(event.frp_valid_count or 0),
        day_detection_count=int(event.day_detection_count or 0),
        night_detection_count=int(event.night_detection_count or 0),
        min_latitude=event.min_latitude,
        max_latitude=event.max_latitude,
        min_longitude=event.min_longitude,
        max_longitude=event.max_longitude,
        is_active=bool(event.is_active),
    )


def _point_geometry(lon: float, lat: float) -> WKTElement:
    return WKTElement(f"POINT({lon} {lat})", srid=4326)


def deactivate_stale_events(
    session: Session,
    reference_time: datetime,
    config: RealtimeEventConfig,
) -> int:
    """Mark active events outside the temporal window as inactive."""
    active = list(
        session.scalars(select(ThermalEvent).where(ThermalEvent.is_active.is_(True))).all()
    )
    states = []
    for ev in active:
        try:
            states.append(thermal_event_to_state(ev))
        except ValueError:
            continue
    to_close = set(events_to_deactivate(states, reference_time, config))
    if not to_close:
        return 0
    now = datetime.now(timezone.utc)
    session.execute(
        update(ThermalEvent)
        .where(ThermalEvent.event_id.in_(to_close))
        .values(is_active=False, updated_at=now)
    )
    return len(to_close)


def _apply_state_to_new_event(session: Session, state: ActiveEventState) -> ThermalEvent:
    now = datetime.now(timezone.utc)
    duration = None
    if state.event_start and state.event_end:
        duration = (state.event_end - state.event_start).total_seconds() / 3600.0
    event = ThermalEvent(
        event_id=state.event_id,
        event_start=state.event_start,
        event_end=state.event_end,
        observed_duration_hours=duration,
        detection_count=state.detection_count,
        peak_frp=state.peak_frp,
        mean_frp=state.mean_frp,
        total_frp=state.total_frp,
        frp_valid_count=state.frp_valid_count,
        day_detection_count=state.day_detection_count,
        night_detection_count=state.night_detection_count,
        centroid_latitude=state.centroid_latitude,
        centroid_longitude=state.centroid_longitude,
        min_latitude=state.min_latitude,
        max_latitude=state.max_latitude,
        min_longitude=state.min_longitude,
        max_longitude=state.max_longitude,
        centroid_wkt=f"POINT ({state.centroid_longitude} {state.centroid_latitude})",
        geometry=_point_geometry(state.centroid_longitude, state.centroid_latitude),
        is_active=True,
        last_detection_at=state.last_detection_at,
        updated_at=now,
    )
    session.add(event)
    session.flush()
    return event


def _apply_state_to_existing_event(event: ThermalEvent, state: ActiveEventState) -> None:
    now = datetime.now(timezone.utc)
    event.event_start = state.event_start
    event.event_end = state.event_end
    if state.event_start and state.event_end:
        event.observed_duration_hours = (
            state.event_end - state.event_start
        ).total_seconds() / 3600.0
    event.detection_count = state.detection_count
    event.peak_frp = state.peak_frp
    event.mean_frp = state.mean_frp
    event.total_frp = state.total_frp
    event.frp_valid_count = state.frp_valid_count
    event.day_detection_count = state.day_detection_count
    event.night_detection_count = state.night_detection_count
    event.centroid_latitude = state.centroid_latitude
    event.centroid_longitude = state.centroid_longitude
    event.min_latitude = state.min_latitude
    event.max_latitude = state.max_latitude
    event.min_longitude = state.min_longitude
    event.max_longitude = state.max_longitude
    event.centroid_wkt = f"POINT ({state.centroid_longitude} {state.centroid_latitude})"
    event.geometry = _point_geometry(state.centroid_longitude, state.centroid_latitude)
    event.is_active = True
    event.last_detection_at = state.last_detection_at
    event.updated_at = now


def process_one_observation(
    session: Session,
    obs: FirmsObservation,
    *,
    config: Optional[RealtimeEventConfig] = None,
) -> MatchAction:
    """
    Transactional single-observation processing (caller may wrap larger txn).

    Idempotent if obs.event_id is already set or event_detections exists.
    """
    cfg = config or default_realtime_config()

    if obs.event_id is not None:
        return MatchAction.SKIPPED_ALREADY_ASSIGNED

    existing_link = session.scalar(
        select(EventDetection.id).where(
            EventDetection.observation_hash == obs.observation_hash
        )
    )
    if existing_link is not None:
        return MatchAction.SKIPPED_ALREADY_ASSIGNED

    try:
        record = observation_to_record(obs)
    except ValueError:
        return MatchAction.SKIPPED_INVALID

    deactivate_stale_events(session, record.acq_datetime, cfg)

    active_rows = list(
        session.scalars(
            select(ThermalEvent).where(
                ThermalEvent.is_active.is_(True),
                ThermalEvent.centroid_latitude.is_not(None),
                ThermalEvent.centroid_longitude.is_not(None),
            )
        ).all()
    )
    active_states = []
    for row in active_rows:
        try:
            active_states.append(thermal_event_to_state(row))
        except ValueError:
            continue

    # Pre-allocate ID only if needed after match attempt — but matcher needs
    # the ID for create path. Probe match first via process_observation with
    # a placeholder only when creating.
    from realtime.event_matcher import match_observation_to_event

    matched = match_observation_to_event(record, active_states, cfg)
    new_id: Optional[str] = None
    if matched is None:
        new_id = allocate_event_id(session)

    result = process_observation(
        record,
        active_states,
        new_event_id=new_id,
        config=cfg,
    )

    if result.action in {
        MatchAction.SKIPPED_ALREADY_ASSIGNED,
        MatchAction.SKIPPED_INVALID,
    }:
        return result.action

    assert result.updated_event is not None
    state = result.updated_event

    if result.action == MatchAction.CREATED:
        _apply_state_to_new_event(session, state)
    else:
        event = session.scalar(
            select(ThermalEvent).where(ThermalEvent.event_id == state.event_id)
        )
        if event is None:
            raise RuntimeError(f"Matched event missing in DB: {state.event_id}")
        _apply_state_to_existing_event(event, state)

    session.add(
        EventDetection(
            event_id=state.event_id,
            observation_hash=obs.observation_hash,
        )
    )
    obs.event_id = state.event_id
    session.flush()
    return result.action


def process_unassigned_observations(
    session: Session,
    *,
    observation_hashes: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    config: Optional[RealtimeEventConfig] = None,
    commit: bool = True,
) -> EventFormationStats:
    """
    Process FIRMS observations with NULL event_id through Phase 3.

    Each observation is handled in the same session; commit once at the end
    when ``commit=True``. Per-observation failures roll back only if the
    caller wraps differently — here we process sequentially and commit as a
    batch for the poll cycle.
    """
    cfg = config or default_realtime_config()
    stats = EventFormationStats()

    stmt = select(FirmsObservation).where(FirmsObservation.event_id.is_(None))
    if observation_hashes is not None:
        stmt = stmt.where(FirmsObservation.observation_hash.in_(list(observation_hashes)))
    stmt = stmt.order_by(
        FirmsObservation.acq_datetime.asc().nullslast(),
        FirmsObservation.id.asc(),
    )
    if limit is not None:
        stmt = stmt.limit(limit)

    observations = list(session.scalars(stmt).all())
    for obs in observations:
        action = process_one_observation(session, obs, config=cfg)
        stats.processed += 1
        if action == MatchAction.CREATED:
            stats.created += 1
            if obs.event_id:
                stats.event_ids_touched.append(obs.event_id)
        elif action == MatchAction.MATCHED:
            stats.matched += 1
            if obs.event_id:
                stats.event_ids_touched.append(obs.event_id)
        elif action == MatchAction.SKIPPED_ALREADY_ASSIGNED:
            stats.skipped_already_assigned += 1
        else:
            stats.skipped_invalid += 1

    if commit:
        session.commit()
    else:
        session.flush()

    logger.info(
        "Phase 3 event formation: processed=%s created=%s matched=%s skipped=%s",
        stats.processed,
        stats.created,
        stats.matched,
        stats.skipped_already_assigned + stats.skipped_invalid,
    )
    return stats
