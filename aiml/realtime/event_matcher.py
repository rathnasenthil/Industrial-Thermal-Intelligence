"""
Incremental event matcher.

Uses the same continuity parameters as Stage G ST-DBSCAN
(``spatial_eps_km``, ``temporal_eps_hours``) but does **not** re-cluster
the full history. Matching is against active event centroids and
``last_detection_at`` so historical ``event_id`` values remain stable.

Selection among eligible candidates is deterministic:
1. smallest temporal gap to last_detection_at
2. smallest great-circle distance to centroid
3. lexicographically smallest event_id as final tie-breaker

Facility proximity, risk score, and anomaly score are never used here.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional, Sequence

from .config import RealtimeEventConfig, default_realtime_config
from .schemas import ActiveEventState, ObservationRecord


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres (WGS84 sphere approximation)."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def temporal_gap_hours(a: datetime, b: datetime) -> float:
    return abs((_ensure_utc(a) - _ensure_utc(b)).total_seconds()) / 3600.0


def is_event_temporally_eligible(
    event: ActiveEventState,
    observation_time: datetime,
    config: RealtimeEventConfig,
) -> bool:
    if event.last_detection_at is None:
        return False
    return (
        temporal_gap_hours(observation_time, event.last_detection_at)
        <= config.temporal_eps_hours
    )


def is_event_spatially_eligible(
    event: ActiveEventState,
    latitude: float,
    longitude: float,
    config: RealtimeEventConfig,
) -> bool:
    distance = haversine_km(
        latitude, longitude, event.centroid_latitude, event.centroid_longitude
    )
    return distance <= config.spatial_eps_km


def match_observation_to_event(
    observation: ObservationRecord,
    active_events: Sequence[ActiveEventState],
    config: Optional[RealtimeEventConfig] = None,
) -> Optional[tuple[ActiveEventState, float, float]]:
    """
    Return (best_event, temporal_gap_hours, distance_km) or None.

    Only considers events that are flagged active and within spatial +
    temporal continuity thresholds.
    """
    cfg = config or default_realtime_config()
    candidates: list[tuple[float, float, str, ActiveEventState]] = []

    for event in active_events:
        if not event.is_active:
            continue
        if not is_event_temporally_eligible(event, observation.acq_datetime, cfg):
            continue
        if not is_event_spatially_eligible(
            event, observation.latitude, observation.longitude, cfg
        ):
            continue
        t_gap = temporal_gap_hours(observation.acq_datetime, event.last_detection_at)
        dist = haversine_km(
            observation.latitude,
            observation.longitude,
            event.centroid_latitude,
            event.centroid_longitude,
        )
        candidates.append((t_gap, dist, event.event_id, event))

    if not candidates:
        return None

    candidates.sort(key=lambda row: (row[0], row[1], row[2]))
    t_gap, dist, _, best = candidates[0]
    return best, t_gap, dist


def events_to_deactivate(
    active_events: Sequence[ActiveEventState],
    reference_time: datetime,
    config: Optional[RealtimeEventConfig] = None,
) -> list[str]:
    """Event IDs whose last_detection_at is outside the temporal window."""
    cfg = config or default_realtime_config()
    out: list[str] = []
    for event in active_events:
        if not event.is_active:
            continue
        if not is_event_temporally_eligible(event, reference_time, cfg):
            out.append(event.event_id)
    return out
