"""Plain data structures for incremental event formation (no ORM)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class MatchAction(str, Enum):
    CREATED = "created"
    MATCHED = "matched"
    SKIPPED_ALREADY_ASSIGNED = "skipped_already_assigned"
    SKIPPED_INVALID = "skipped_invalid"


@dataclass
class ObservationRecord:
    """One FIRMS observation ready for incremental event formation."""

    observation_hash: str
    latitude: float
    longitude: float
    acq_datetime: datetime
    frp: Optional[float] = None
    bright_ti4: Optional[float] = None
    bright_ti5: Optional[float] = None
    daynight: Optional[str] = None
    confidence: Optional[str] = None
    event_id: Optional[str] = None  # non-null ⇒ already assigned

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_hash": self.observation_hash,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "acq_datetime": self.acq_datetime,
            "frp": self.frp,
            "bright_ti4": self.bright_ti4,
            "bright_ti5": self.bright_ti5,
            "daynight": self.daynight,
            "confidence": self.confidence,
            "event_id": self.event_id,
        }


@dataclass
class ActiveEventState:
    """
    Minimal active-event snapshot for matching.

    ``is_active`` means the event is still within the temporal continuity
    window for accepting new observations — not that it is a confirmed fire.
    """

    event_id: str
    centroid_latitude: float
    centroid_longitude: float
    last_detection_at: datetime
    event_start: Optional[datetime] = None
    event_end: Optional[datetime] = None
    detection_count: int = 0
    peak_frp: Optional[float] = None
    mean_frp: Optional[float] = None
    total_frp: Optional[float] = None
    frp_valid_count: int = 0
    day_detection_count: int = 0
    night_detection_count: int = 0
    min_latitude: Optional[float] = None
    max_latitude: Optional[float] = None
    min_longitude: Optional[float] = None
    max_longitude: Optional[float] = None
    is_active: bool = True
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessResult:
    event_id: Optional[str]
    action: MatchAction
    matched_existing_event: bool
    observation_hash: str
    updated_event: Optional[ActiveEventState] = None
    temporal_gap_hours: Optional[float] = None
    spatial_distance_km: Optional[float] = None
    reason: Optional[str] = None
