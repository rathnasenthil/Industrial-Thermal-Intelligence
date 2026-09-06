"""
Incremental thermal-event aggregate updates.

Mirrors Stage G ``build_thermal_events`` / ``compute_event_row`` semantics
for the fields maintained in realtime:

- event_start / event_end = min / max observation times
- detection_count increments by 1
- peak_frp = max of valid FRP values
- mean_frp = total_frp / frp_valid_count (not a recursive average of averages)
- centroid = arithmetic mean of member coordinates (same as batch)
- bbox min/max latitudes/longitudes expand to include the new point

Null FRP does not contribute to peak/mean/total; the observation still
increments detection_count.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Optional

from .schemas import ActiveEventState, ObservationRecord


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _frp_is_valid(frp: Optional[float]) -> bool:
    return frp is not None and frp == frp  # not None and not NaN


def create_event_state_from_observation(
    event_id: str,
    observation: ObservationRecord,
) -> ActiveEventState:
    """Initialize a new event from its first observation."""
    ts = _ensure_utc(observation.acq_datetime)
    frp = observation.frp if _frp_is_valid(observation.frp) else None
    day = 1 if (observation.daynight or "").upper() == "D" else 0
    night = 1 if (observation.daynight or "").upper() == "N" else 0
    return ActiveEventState(
        event_id=event_id,
        centroid_latitude=float(observation.latitude),
        centroid_longitude=float(observation.longitude),
        last_detection_at=ts,
        event_start=ts,
        event_end=ts,
        detection_count=1,
        peak_frp=frp,
        mean_frp=frp,
        total_frp=frp if frp is not None else 0.0,
        frp_valid_count=1 if frp is not None else 0,
        day_detection_count=day,
        night_detection_count=night,
        min_latitude=float(observation.latitude),
        max_latitude=float(observation.latitude),
        min_longitude=float(observation.longitude),
        max_longitude=float(observation.longitude),
        is_active=True,
    )


def update_event_with_observation(
    event: ActiveEventState,
    observation: ObservationRecord,
) -> ActiveEventState:
    """
    Return an updated copy of ``event`` after attaching ``observation``.

    Mean FRP uses a running sum / valid count (``total_frp`` / ``frp_valid_count``),
    never ``(old_mean + new) / 2``.
    """
    updated = deepcopy(event)
    ts = _ensure_utc(observation.acq_datetime)
    n_prev = max(int(updated.detection_count or 0), 0)
    n_new = n_prev + 1

    # Running mean centroid (batch uses arithmetic mean of all points).
    updated.centroid_latitude = (
        (updated.centroid_latitude * n_prev) + float(observation.latitude)
    ) / n_new
    updated.centroid_longitude = (
        (updated.centroid_longitude * n_prev) + float(observation.longitude)
    ) / n_new

    updated.min_latitude = min(
        updated.min_latitude if updated.min_latitude is not None else observation.latitude,
        float(observation.latitude),
    )
    updated.max_latitude = max(
        updated.max_latitude if updated.max_latitude is not None else observation.latitude,
        float(observation.latitude),
    )
    updated.min_longitude = min(
        updated.min_longitude if updated.min_longitude is not None else observation.longitude,
        float(observation.longitude),
    )
    updated.max_longitude = max(
        updated.max_longitude if updated.max_longitude is not None else observation.longitude,
        float(observation.longitude),
    )

    start = updated.event_start or ts
    end = updated.event_end or ts
    updated.event_start = min(_ensure_utc(start), ts)
    updated.event_end = max(_ensure_utc(end), ts)
    last = updated.last_detection_at or ts
    updated.last_detection_at = max(_ensure_utc(last), ts)
    updated.detection_count = n_new

    if _frp_is_valid(observation.frp):
        frp = float(observation.frp)  # type: ignore[arg-type]
        prev_total = float(updated.total_frp or 0.0)
        prev_valid = int(updated.frp_valid_count or 0)
        new_total = prev_total + frp
        new_valid = prev_valid + 1
        updated.total_frp = new_total
        updated.frp_valid_count = new_valid
        updated.mean_frp = new_total / new_valid
        updated.peak_frp = (
            frp if updated.peak_frp is None else max(float(updated.peak_frp), frp)
        )

    if (observation.daynight or "").upper() == "D":
        updated.day_detection_count = int(updated.day_detection_count or 0) + 1
    elif (observation.daynight or "").upper() == "N":
        updated.night_detection_count = int(updated.night_detection_count or 0) + 1

    duration_h = (
        updated.event_end - updated.event_start
    ).total_seconds() / 3600.0
    updated.extras["observed_duration_hours"] = duration_h
    updated.is_active = True
    return updated
