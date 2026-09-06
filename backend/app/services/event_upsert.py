"""
Backend adapter: incremental thermal-event formation (Phase 3), G.1 (Phase 4),
I.2 facility association (Phase 5), I.3 facility fingerprinting (Phase 6),
I.4 temporal anomaly detection (Phase 7), I.5 STA evidence (Phase 8),
I.6 environmental context (Phase 9), I.7 evidence fusion (Phase 10),
and Stage VI risk prioritization (Phase 11).

Translates DB rows ↔ AIML ``realtime`` plain objects, allocates stable
``EVT_#######`` IDs from a PostgreSQL sequence, and commits observation
linkage transactionally.

Phase 4 recomputes Stage G.1 persistence fields for the *affected event only*.
Phase 5 recomputes Stage I.2 facility association for the *affected event only*
(spatial attribution — not source classification).
Phase 6 recomputes Stage I.3 fingerprints for *affected facilities only*
(descriptive baseline — not anomaly detection).
Phase 7 recomputes Stage I.4 walk-forward anomaly for the *affected event only*
(prior-only temporal deviation — not risk / fire classification).
Phase 8 recomputes Stage I.5 STA evidence for the *affected event only*
(supporting evidence — not ground truth / industrial-fire classification).
Phase 9 recomputes Stage I.6 environmental context for the *affected event only*
(context/evidence — not classification / risk).
Phase 10 recomputes Stage I.7 evidence fusion for the *affected event only*
(interpretation — not ground truth / risk probability).
Phase 11 recomputes Stage VI risk prioritization for the *affected event only*
(decision-support score — not fire probability).

Does not run ST-DBSCAN batch clustering, full-table
``run_persistence_characterization``, full-table ``run_facility_association``,
full-table ``run_facility_fingerprinting``, full-table ``run_anomaly_detection``,
full-table ``run_sta_integration``, full-table ``run_environmental_context``,
full-table ``run_evidence_fusion``, or full-table ``run_risk_prioritization``.
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
from app.models.event_facility_candidate import EventFacilityCandidate
from app.models.facility_thermal_fingerprint import FacilityThermalFingerprint
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
from realtime.persistence import PersistenceFeatures, process_event_persistence  # noqa: E402
from realtime.schemas import (  # noqa: E402
    ActiveEventState,
    MatchAction,
    ObservationRecord,
)
from app.services.facility_association import (  # noqa: E402
    refresh_event_facility_association,
)
from app.services.anomaly import refresh_event_anomaly  # noqa: E402
from app.services.facility_fingerprint import (  # noqa: E402
    refresh_fingerprints_for_event,
)
from app.services.sta import refresh_event_sta  # noqa: E402
from app.services.environmental import refresh_event_environmental  # noqa: E402
from app.services.evidence_fusion import refresh_event_evidence_fusion  # noqa: E402
from app.services.risk import refresh_event_risk  # noqa: E402


@dataclass
class EventFormationStats:
    processed: int = 0
    created: int = 0
    matched: int = 0
    skipped_already_assigned: int = 0
    skipped_invalid: int = 0
    deactivated: int = 0
    event_ids_touched: list[str] = field(default_factory=list)
    # Phase 4
    persistence_updated: int = 0
    persistence_unchanged: int = 0
    persistence_by_event: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Phase 5
    facility_association_updated: int = 0
    facility_association_by_event: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Phase 6
    fingerprint_facilities_refreshed: int = 0
    fingerprint_by_facility: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Phase 7
    anomaly_updated: int = 0
    anomaly_by_event: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Phase 8
    sta_updated: int = 0
    sta_by_event: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Phase 9
    environmental_updated: int = 0
    environmental_by_event: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Phase 10
    fusion_updated: int = 0
    fusion_by_event: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Phase 11
    risk_updated: int = 0
    risk_by_event: dict[str, dict[str, Any]] = field(default_factory=dict)

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


def load_event_detection_datetimes(session: Session, event_id: str) -> list[datetime]:
    """
    Load FIRMS acquisition times for one event via ``event_detections``.

    Plain datetimes only — AIML persistence must not depend on ORM rows.
    """
    rows = session.execute(
        select(FirmsObservation.acq_datetime)
        .join(
            EventDetection,
            EventDetection.observation_hash == FirmsObservation.observation_hash,
        )
        .where(EventDetection.event_id == event_id)
        .order_by(FirmsObservation.acq_datetime.asc().nullslast())
    ).all()
    out: list[datetime] = []
    for (acq,) in rows:
        if acq is None:
            continue
        out.append(_ensure_utc(acq))
    return out


def _persistence_snapshot(event: ThermalEvent) -> tuple:
    """Comparable tuple of G.1 fields currently stored on the event."""
    return (
        event.detection_count,
        event.distinct_detection_days,
        event.span_days,
        event.observed_duration_hours,
        event.duty_cycle,
        event.mean_gap_hours,
        event.max_gap_hours,
        event.persistence_label,
        event.persistence_basis,
    )


def apply_persistence_features(event: ThermalEvent, features: PersistenceFeatures) -> bool:
    """
    Write G.1 fields onto an existing ThermalEvent row.

    Reuses Stage VII columns; does not create schema. Returns True if any
    tracked persistence field changed.
    """
    before = _persistence_snapshot(event)
    event.detection_count = features.detection_count
    event.distinct_detection_days = features.distinct_detection_days
    event.span_days = float(features.span_days)
    event.observed_duration_hours = features.observed_duration_hours
    event.duty_cycle = features.duty_cycle
    event.mean_gap_hours = features.mean_gap_hours
    event.max_gap_hours = features.max_gap_hours
    event.persistence_label = features.persistence_label
    event.persistence_basis = features.persistence_basis
    # Keep temporal anchors consistent with the detection set used for G.1.
    event.event_start = features.event_start
    event.event_end = features.event_end
    event.updated_at = datetime.now(timezone.utc)
    after = _persistence_snapshot(event)
    return before != after


def refresh_event_persistence(
    session: Session,
    event_id: str,
) -> PersistenceFeatures:
    """
    Phase 4: recompute G.1 for one event from its linked detections.

    Does **not** call ``run_persistence_characterization()`` over all events.
    """
    times = load_event_detection_datetimes(session, event_id)
    if not times:
        raise ValueError(f"event {event_id} has no linked detection timestamps")
    features = process_event_persistence(event_id, times)
    event = session.scalar(
        select(ThermalEvent).where(ThermalEvent.event_id == event_id)
    )
    if event is None:
        raise RuntimeError(f"ThermalEvent missing for persistence update: {event_id}")
    apply_persistence_features(event, features)
    session.flush()
    return features


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

    previous_facility_id: Optional[str] = None
    if result.action == MatchAction.CREATED:
        _apply_state_to_new_event(session, state)
    else:
        event = session.scalar(
            select(ThermalEvent).where(ThermalEvent.event_id == state.event_id)
        )
        if event is None:
            raise RuntimeError(f"Matched event missing in DB: {state.event_id}")
        previous_facility_id = event.facility_id
        _apply_state_to_existing_event(event, state)

    session.add(
        EventDetection(
            event_id=state.event_id,
            observation_hash=obs.observation_hash,
        )
    )
    obs.event_id = state.event_id
    session.flush()

    # Phase 4: G.1 for the affected event only (same transaction).
    refresh_event_persistence(session, state.event_id)
    # Phase 5: I.2 facility association for the affected event only.
    refresh_event_facility_association(session, state.event_id)
    # Phase 6: I.3 fingerprint for affected facilities only.
    refresh_fingerprints_for_event(
        session,
        state.event_id,
        previous_facility_id=previous_facility_id,
    )
    # Phase 7: I.4 walk-forward anomaly for the affected event only.
    refresh_event_anomaly(session, state.event_id)
    # Phase 8: I.5 STA evidence for the affected event only.
    refresh_event_sta(session, state.event_id)
    # Phase 9: I.6 environmental context for the affected event only.
    refresh_event_environmental(session, state.event_id)
    # Phase 10: I.7 evidence fusion for the affected event only.
    refresh_event_evidence_fusion(session, state.event_id)
    # Phase 11: Stage VI risk prioritization for the affected event only.
    refresh_event_risk(session, state.event_id)
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
    Process FIRMS observations with NULL event_id through Phases 3–11.

    Phase 3 attaches detections; Phase 4 recomputes G.1; Phase 5 runs I.2;
    Phase 6 refreshes I.3 fingerprints for affected facilities only;
    Phase 7 scores I.4 for each affected event only;
    Phase 8 attaches I.5 STA evidence for each affected event only;
    Phase 9 attaches I.6 environmental context for each affected event only;
    Phase 10 fuses I.7 evidence for each affected event only;
    Phase 11 scores Stage VI risk for each affected event only.
    Does not run batch orchestrators over historical tables.
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

        if obs.event_id and action in {MatchAction.CREATED, MatchAction.MATCHED}:
            event = session.scalar(
                select(ThermalEvent).where(ThermalEvent.event_id == obs.event_id)
            )
            if event is not None:
                stats.persistence_by_event[obs.event_id] = {
                    "detection_count": event.detection_count,
                    "distinct_detection_days": event.distinct_detection_days,
                    "span_days": event.span_days,
                    "observed_duration_hours": event.observed_duration_hours,
                    "duty_cycle": event.duty_cycle,
                    "mean_gap_hours": event.mean_gap_hours,
                    "max_gap_hours": event.max_gap_hours,
                    "persistence_label": event.persistence_label,
                }
                stats.facility_association_by_event[obs.event_id] = {
                    "facility_id": event.facility_id,
                    "facility_association_method": event.facility_association_method,
                    "facility_attribution_confidence": event.facility_attribution_confidence,
                    "facility_distance_km": event.facility_distance_km,
                    "candidate_facility_count": event.candidate_facility_count,
                }
                fp_ids: list[str] = []
                if event.facility_id:
                    fp_ids.append(event.facility_id)
                elif event.facility_association_method == "AMBIGUOUS":
                    fp_ids.extend(
                        session.scalars(
                            select(EventFacilityCandidate.facility_id).where(
                                EventFacilityCandidate.event_id == event.event_id
                            )
                        ).all()
                    )
                for fid in fp_ids:
                    if not fid or fid in stats.fingerprint_by_facility:
                        continue
                    fp = session.scalar(
                        select(FacilityThermalFingerprint).where(
                            FacilityThermalFingerprint.facility_id == fid
                        )
                    )
                    if fp is not None:
                        stats.fingerprint_by_facility[fid] = {
                            "event_count": fp.event_count,
                            "fingerprint_status": fp.fingerprint_status,
                            "ambiguous_candidate_opportunity_count": (
                                fp.ambiguous_candidate_opportunity_count
                            ),
                        }
                stats.anomaly_by_event[obs.event_id] = {
                    "anomaly_score": event.anomaly_score,
                    "anomaly_status": event.anomaly_status,
                    "anomaly_confidence": event.anomaly_confidence,
                    "baseline_observation_count": event.baseline_observation_count,
                    "baseline_history_status": event.baseline_history_status,
                    "anomaly_unavailable_reason": event.anomaly_unavailable_reason,
                }
                stats.sta_by_event[obs.event_id] = {
                    "sta_association_status": event.sta_association_status,
                    "sta_evidence_available": event.sta_evidence_available,
                    "sta_evidence_quality": event.sta_evidence_quality,
                    "sta_match_count": event.sta_match_count,
                    "primary_sta_id": event.primary_sta_id,
                }
                stats.environmental_by_event[obs.event_id] = {
                    "landcover_available": event.landcover_available,
                    "vegetation_context_available": event.vegetation_context_available,
                    "builtup_context_available": event.builtup_context_available,
                    "water_context_available": event.water_context_available,
                    "agriculture_context_available": event.agriculture_context_available,
                    "satellite_context_available": event.satellite_context_available,
                    "dominant_landcover_class": event.dominant_landcover_class,
                    "water_present": event.water_present,
                }
                stats.fusion_by_event[obs.event_id] = {
                    "source_intelligence_candidate": event.source_intelligence_candidate,
                    "evidence_strength": event.evidence_strength,
                    "evidence_fusion_score": event.evidence_fusion_score,
                    "evidence_sufficiency": event.evidence_sufficiency,
                    "candidate_is_ground_truth": event.candidate_is_ground_truth,
                }
                stats.risk_by_event[obs.event_id] = {
                    "risk_score": event.risk_score,
                    "investigation_priority": event.investigation_priority,
                    "industrial_context": event.industrial_context,
                    "recommended_action": event.recommended_action,
                }

    unique_touched = sorted(set(stats.event_ids_touched))
    stats.persistence_updated = len(unique_touched)
    stats.facility_association_updated = len(unique_touched)
    stats.fingerprint_facilities_refreshed = len(stats.fingerprint_by_facility)
    stats.anomaly_updated = len(stats.anomaly_by_event)
    stats.sta_updated = len(stats.sta_by_event)
    stats.environmental_updated = len(stats.environmental_by_event)
    stats.fusion_updated = len(stats.fusion_by_event)
    stats.risk_updated = len(stats.risk_by_event)
    stats.persistence_unchanged = 0

    if commit:
        session.commit()
    else:
        session.flush()

    logger.info(
        "Phase 3-11 formation/.../risk: processed=%s created=%s matched=%s "
        "anomaly=%s sta=%s env=%s fusion=%s risk=%s skipped=%s",
        stats.processed,
        stats.created,
        stats.matched,
        stats.anomaly_updated,
        stats.sta_updated,
        stats.environmental_updated,
        stats.fusion_updated,
        stats.risk_updated,
        stats.skipped_already_assigned + stats.skipped_invalid,
    )
    return stats
