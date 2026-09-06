"""Database tests for Phase 3 incremental event formation."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from geoalchemy2.elements import WKTElement
from sqlalchemy import func, select, text

AIML_ROOT = Path(__file__).resolve().parents[2] / "aiml"
if str(AIML_ROOT) not in sys.path:
    sys.path.insert(0, str(AIML_ROOT))

from realtime.schemas import MatchAction

from app.models.event_detection import EventDetection
from app.models.firms_observation import FirmsObservation
from app.models.thermal_event import ThermalEvent
from app.services.event_upsert import (
    allocate_event_id,
    process_one_observation,
    process_unassigned_observations,
)
from app.services.observation_identity import compute_observation_hash
from tests.conftest import REQUIRES_POSTGIS

MARKER = "phase3_test_sat"


def _hash_row(**overrides) -> dict:
    row = {
        "latitude": "20.00000",
        "longitude": "85.00000",
        "bright_ti4": "330",
        "scan": "0.4",
        "track": "0.4",
        "acq_date": "2026-09-05",
        "acq_time": "0600",
        "satellite": MARKER,
        "instrument": "VIIRS",
        "confidence": "n",
        "version": "2.0NRT",
        "bright_ti5": "290",
        "frp": "2.0",
        "daynight": "D",
        "type": "0",
    }
    row.update(overrides)
    return row


def _insert_obs(session, *, lat, lon, acq_time, frp="2.0", acq_date="2026-09-05") -> FirmsObservation:
    raw = _hash_row(
        latitude=f"{lat:.5f}",
        longitude=f"{lon:.5f}",
        acq_time=acq_time,
        frp=str(frp),
        acq_date=acq_date,
    )
    # uniquify type/frp fields slightly via acq_time already in hash
    obs_hash = compute_observation_hash(raw)
    hours = int(acq_time[:2])
    minutes = int(acq_time[2:])
    ts = datetime(2026, 9, 5, hours, minutes, tzinfo=timezone.utc)
    obs = FirmsObservation(
        observation_hash=obs_hash,
        latitude=float(lat),
        longitude=float(lon),
        geometry=WKTElement(f"POINT({lon} {lat})", srid=4326),
        acq_date=acq_date,
        acq_time=acq_time,
        acq_datetime=ts,
        satellite=MARKER,
        instrument="VIIRS",
        confidence="n",
        version="2.0NRT",
        bright_ti4=330.0,
        bright_ti5=290.0,
        scan=0.4,
        track=0.4,
        frp=float(frp),
        daynight="D",
        type="0",
        source_file="phase3_test",
        event_id=None,
    )
    session.add(obs)
    session.flush()
    return obs


def _cleanup(session) -> None:
    session.execute(
        text(
            "DELETE FROM event_detections WHERE observation_hash IN "
            "(SELECT observation_hash FROM firms_observations WHERE satellite = :s)"
        ),
        {"s": MARKER},
    )
    session.execute(
        text(
            "DELETE FROM thermal_events WHERE event_id IN "
            "(SELECT DISTINCT event_id FROM firms_observations "
            " WHERE satellite = :s AND event_id IS NOT NULL)"
        ),
        {"s": MARKER},
    )
    # Also delete NRT events created in tests that might not be linked if failed mid-way
    session.execute(
        text("DELETE FROM firms_observations WHERE satellite = :s"),
        {"s": MARKER},
    )
    session.commit()


@REQUIRES_POSTGIS
def test_db_create_match_idempotent_and_aggregates(db_session) -> None:
    _cleanup(db_session)

    hist_before = db_session.scalar(select(func.count()).select_from(ThermalEvent))

    o1 = _insert_obs(db_session, lat=20.0, lon=85.0, acq_time="0600", frp="2.0")
    o2 = _insert_obs(db_session, lat=20.004, lon=85.0, acq_time="0700", frp="6.0")
    db_session.commit()

    stats = process_unassigned_observations(
        db_session,
        observation_hashes=[o1.observation_hash, o2.observation_hash],
        commit=True,
    )
    assert stats.created == 1
    assert stats.matched == 1
    assert stats.processed == 2

    db_session.refresh(o1)
    db_session.refresh(o2)
    assert o1.event_id is not None
    assert o1.event_id == o2.event_id
    assert o1.event_id.startswith("EVT_")

    event = db_session.scalar(
        select(ThermalEvent).where(ThermalEvent.event_id == o1.event_id)
    )
    assert event is not None
    assert event.is_active is True
    assert event.detection_count == 2
    assert event.peak_frp == pytest.approx(6.0)
    assert event.mean_frp == pytest.approx(4.0)
    assert event.event_start <= event.event_end

    det_count = db_session.scalar(
        select(func.count()).where(EventDetection.event_id == o1.event_id)
    )
    assert det_count == 2

    # Idempotent re-process
    stats2 = process_unassigned_observations(
        db_session,
        observation_hashes=[o1.observation_hash, o2.observation_hash],
        commit=True,
    )
    assert stats2.processed == 0  # no unassigned left among these hashes
    det_count2 = db_session.scalar(
        select(func.count()).where(EventDetection.event_id == o1.event_id)
    )
    assert det_count2 == 2

    # Duplicate detection insert blocked by unique constraint path (already assigned)
    action = process_one_observation(db_session, o1)
    assert action == MatchAction.SKIPPED_ALREADY_ASSIGNED

    hist_after = db_session.scalar(select(func.count()).select_from(ThermalEvent))
    # Exactly one new event added on top of historical corpus
    assert hist_after == hist_before + 1

    _cleanup(db_session)


@REQUIRES_POSTGIS
def test_spatial_split_creates_two_events(db_session) -> None:
    _cleanup(db_session)
    o1 = _insert_obs(db_session, lat=20.0, lon=85.0, acq_time="0600")
    o2 = _insert_obs(db_session, lat=21.0, lon=85.0, acq_time="0610")  # ~111 km
    db_session.commit()
    stats = process_unassigned_observations(
        db_session,
        observation_hashes=[o1.observation_hash, o2.observation_hash],
        commit=True,
    )
    assert stats.created == 2
    assert stats.matched == 0
    _cleanup(db_session)


@REQUIRES_POSTGIS
def test_temporal_split_and_inactive_lifecycle(db_session) -> None:
    _cleanup(db_session)
    o1 = _insert_obs(db_session, lat=20.0, lon=85.0, acq_time="0600")
    db_session.commit()
    process_unassigned_observations(
        db_session,
        observation_hashes=[o1.observation_hash],
        commit=True,
    )
    db_session.refresh(o1)
    eid = o1.event_id
    assert eid

    # Force last_detection_at far in the past then process a new nearby obs
    event = db_session.scalar(select(ThermalEvent).where(ThermalEvent.event_id == eid))
    assert event is not None
    event.last_detection_at = datetime(2026, 9, 5, 6, 0, tzinfo=timezone.utc) - timedelta(
        hours=48
    )
    event.is_active = True
    db_session.commit()

    o2 = _insert_obs(db_session, lat=20.001, lon=85.0, acq_time="0800")
    db_session.commit()
    stats = process_unassigned_observations(
        db_session,
        observation_hashes=[o2.observation_hash],
        commit=True,
    )
    assert stats.created == 1
    db_session.refresh(o2)
    assert o2.event_id != eid

    old = db_session.scalar(select(ThermalEvent).where(ThermalEvent.event_id == eid))
    assert old is not None
    assert old.is_active is False

    _cleanup(db_session)


@REQUIRES_POSTGIS
def test_allocate_event_id_format_and_historical_preserved(db_session) -> None:
    sample = db_session.scalar(select(ThermalEvent.event_id).limit(1))
    assert sample is not None
    # Historical IDs still look like EVT_
    assert sample.startswith("EVT_")
    new_id = allocate_event_id(db_session)
    assert new_id.startswith("EVT_")
    assert new_id != sample
    db_session.rollback()


@REQUIRES_POSTGIS
def test_event_detection_unique_constraint(db_session) -> None:
    _cleanup(db_session)
    o1 = _insert_obs(db_session, lat=22.0, lon=86.0, acq_time="0900")
    db_session.commit()
    process_unassigned_observations(
        db_session,
        observation_hashes=[o1.observation_hash],
        commit=True,
    )
    db_session.refresh(o1)
    with pytest.raises(Exception):
        db_session.add(
            EventDetection(event_id=o1.event_id, observation_hash=o1.observation_hash)
        )
        db_session.commit()
    db_session.rollback()
    _cleanup(db_session)
