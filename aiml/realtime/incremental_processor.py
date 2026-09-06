"""
Framework-independent incremental observation → event processor.

``process_observation`` decides create vs match and returns an updated
event state. Database persistence (SQLAlchemy / PostGIS) lives in the backend.

After an observation is attached, callers should run Phase 4 G.1 via
``realtime.persistence.process_event_persistence`` for **that event only**,
using the event's stored detection timestamps — never
``run_persistence_characterization()`` over the full historical table.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Sequence

from src.persistence.config import PersistenceConfig

from .config import RealtimeEventConfig, default_realtime_config
from .event_matcher import match_observation_to_event
from .feature_updater import (
    create_event_state_from_observation,
    update_event_with_observation,
)
from .persistence import PersistenceFeatures, process_event_persistence
from .schemas import (
    ActiveEventState,
    MatchAction,
    ObservationRecord,
    ProcessResult,
)


def process_observation(
    observation: ObservationRecord,
    active_events: Sequence[ActiveEventState],
    *,
    new_event_id: Optional[str] = None,
    config: Optional[RealtimeEventConfig] = None,
) -> ProcessResult:
    """
    Attach one observation to an existing active event or open a new event.

    Args:
        observation: Observation to process. If ``event_id`` is already set,
            returns SKIPPED_ALREADY_ASSIGNED (idempotent).
        active_events: Currently active events (caller filters / deactivates).
        new_event_id: Required when no match is found — allocated by the
            backend from persisted sequence state (never from row order).
        config: Continuity thresholds (defaults = Stage G STDBSCANConfig).

    Returns:
        ProcessResult with action created|matched|skipped_* and updated state.
    """
    cfg = config or default_realtime_config()

    if not observation.observation_hash:
        return ProcessResult(
            event_id=None,
            action=MatchAction.SKIPPED_INVALID,
            matched_existing_event=False,
            observation_hash="",
            reason="missing observation_hash",
        )

    if observation.event_id:
        return ProcessResult(
            event_id=observation.event_id,
            action=MatchAction.SKIPPED_ALREADY_ASSIGNED,
            matched_existing_event=True,
            observation_hash=observation.observation_hash,
            reason="observation already linked to an event",
        )

    if observation.acq_datetime is None:
        return ProcessResult(
            event_id=None,
            action=MatchAction.SKIPPED_INVALID,
            matched_existing_event=False,
            observation_hash=observation.observation_hash,
            reason="missing acq_datetime",
        )

    matched = match_observation_to_event(observation, active_events, cfg)
    if matched is not None:
        event, t_gap, dist = matched
        updated = update_event_with_observation(event, observation)
        return ProcessResult(
            event_id=updated.event_id,
            action=MatchAction.MATCHED,
            matched_existing_event=True,
            observation_hash=observation.observation_hash,
            updated_event=updated,
            temporal_gap_hours=t_gap,
            spatial_distance_km=dist,
        )

    if not new_event_id:
        return ProcessResult(
            event_id=None,
            action=MatchAction.SKIPPED_INVALID,
            matched_existing_event=False,
            observation_hash=observation.observation_hash,
            reason="no matching active event and new_event_id not provided",
        )

    created = create_event_state_from_observation(new_event_id, observation)
    return ProcessResult(
        event_id=created.event_id,
        action=MatchAction.CREATED,
        matched_existing_event=False,
        observation_hash=observation.observation_hash,
        updated_event=created,
        temporal_gap_hours=None,
        spatial_distance_km=None,
    )


def characterize_event_persistence(
    event_id: str,
    detection_datetimes: Sequence[datetime],
    *,
    config: Optional[PersistenceConfig] = None,
) -> PersistenceFeatures:
    """
    Phase 4 helper: G.1 for one event from its detection times.

    Thin wrapper around ``process_event_persistence`` so orchestration code
    can import formation + G.1 from one module without touching the batch
    persistence orchestrator.
    """
    return process_event_persistence(event_id, detection_datetimes, config=config)
