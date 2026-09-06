"""Unit tests for AIML realtime event matcher / feature updater (no DB)."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

AIML_ROOT = Path(__file__).resolve().parents[2] / "aiml"
if str(AIML_ROOT) not in sys.path:
    sys.path.insert(0, str(AIML_ROOT))

from realtime.config import RealtimeEventConfig
from realtime.feature_updater import (
    create_event_state_from_observation,
    update_event_with_observation,
)
from realtime.incremental_processor import process_observation
from realtime.schemas import ActiveEventState, MatchAction, ObservationRecord


def _ts(hours: float = 0) -> datetime:
    return datetime(2026, 9, 5, 6, 0, tzinfo=timezone.utc) + timedelta(hours=hours)


def _obs(hash_suffix: str, lat: float, lon: float, hours: float, frp: float = 2.0) -> ObservationRecord:
    return ObservationRecord(
        observation_hash=f"hash_{hash_suffix}",
        latitude=lat,
        longitude=lon,
        acq_datetime=_ts(hours),
        frp=frp,
        daynight="D",
    )


def _event(event_id: str, lat: float, lon: float, hours: float, **kwargs) -> ActiveEventState:
    base = ActiveEventState(
        event_id=event_id,
        centroid_latitude=lat,
        centroid_longitude=lon,
        last_detection_at=_ts(hours),
        event_start=_ts(hours),
        event_end=_ts(hours),
        detection_count=1,
        peak_frp=1.0,
        mean_frp=1.0,
        total_frp=1.0,
        frp_valid_count=1,
        is_active=True,
    )
    for k, v in kwargs.items():
        setattr(base, k, v)
    return base


CFG = RealtimeEventConfig(spatial_eps_km=1.5, temporal_eps_hours=36.0)


def test_first_observation_creates_event() -> None:
    result = process_observation(
        _obs("a", 20.0, 85.0, 0),
        [],
        new_event_id="EVT_0000001",
        config=CFG,
    )
    assert result.action == MatchAction.CREATED
    assert result.event_id == "EVT_0000001"
    assert result.matched_existing_event is False
    assert result.updated_event is not None
    assert result.updated_event.detection_count == 1


def test_compatible_observation_matches_existing() -> None:
    existing = _event("EVT_0000001", 20.0, 85.0, 0)
    # ~0.5 km north
    obs = _obs("b", 20.0045, 85.0, 1, frp=4.0)
    result = process_observation(obs, [existing], new_event_id="EVT_X", config=CFG)
    assert result.action == MatchAction.MATCHED
    assert result.event_id == "EVT_0000001"
    assert result.updated_event is not None
    assert result.updated_event.detection_count == 2
    assert result.updated_event.peak_frp == 4.0
    assert result.updated_event.mean_frp == pytest.approx(2.5)
    assert result.updated_event.event_id == "EVT_0000001"


def test_spatial_incompatible_creates_new() -> None:
    existing = _event("EVT_0000001", 20.0, 85.0, 0)
    # ~50 km away
    obs = _obs("c", 20.5, 85.0, 1)
    result = process_observation(obs, [existing], new_event_id="EVT_0000002", config=CFG)
    assert result.action == MatchAction.CREATED
    assert result.event_id == "EVT_0000002"


def test_temporal_incompatible_creates_new() -> None:
    existing = _event("EVT_0000001", 20.0, 85.0, 0)
    obs = _obs("d", 20.001, 85.0, 48)  # 48h > 36h
    result = process_observation(obs, [existing], new_event_id="EVT_0000003", config=CFG)
    assert result.action == MatchAction.CREATED
    assert result.event_id == "EVT_0000003"


def test_inactive_event_not_matched() -> None:
    existing = _event("EVT_0000001", 20.0, 85.0, 0, is_active=False)
    obs = _obs("e", 20.001, 85.0, 1)
    result = process_observation(obs, [existing], new_event_id="EVT_0000004", config=CFG)
    assert result.action == MatchAction.CREATED
    assert result.event_id == "EVT_0000004"


def test_already_assigned_is_idempotent() -> None:
    obs = _obs("f", 20.0, 85.0, 0)
    obs.event_id = "EVT_0000001"
    result = process_observation(obs, [], new_event_id="EVT_X", config=CFG)
    assert result.action == MatchAction.SKIPPED_ALREADY_ASSIGNED


def test_mean_and_peak_frp_and_timestamps() -> None:
    state = create_event_state_from_observation("EVT_1", _obs("g", 20.0, 85.0, 0, frp=2.0))
    assert state.mean_frp == 2.0
    assert state.peak_frp == 2.0
    updated = update_event_with_observation(state, _obs("h", 20.001, 85.0, 2, frp=6.0))
    assert updated.detection_count == 2
    assert updated.peak_frp == 6.0
    assert updated.mean_frp == pytest.approx(4.0)
    assert updated.total_frp == pytest.approx(8.0)
    assert updated.frp_valid_count == 2
    assert updated.event_start == _ts(0)
    assert updated.event_end == _ts(2)
    # Null FRP does not break mean
    updated2 = update_event_with_observation(updated, _obs("i", 20.002, 85.0, 3, frp=None))  # type: ignore[arg-type]
    assert updated2.frp_valid_count == 2
    assert updated2.mean_frp == pytest.approx(4.0)
    assert updated2.detection_count == 3


def test_deterministic_tie_break_prefers_smaller_event_id() -> None:
    # Two events at same place/time — smaller event_id wins after equal gaps/dist.
    e1 = _event("EVT_0000010", 20.0, 85.0, 0)
    e2 = _event("EVT_0000002", 20.0, 85.0, 0)
    obs = _obs("j", 20.0, 85.0, 0.5)
    result = process_observation(obs, [e1, e2], new_event_id="EVT_X", config=CFG)
    assert result.action == MatchAction.MATCHED
    assert result.event_id == "EVT_0000002"
