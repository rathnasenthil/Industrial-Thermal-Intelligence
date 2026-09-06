"""Database tests for Phase 4 incremental G.1 persistence."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from geoalchemy2.elements import WKTElement
from sqlalchemy import select, text
from sqlalchemy.orm import sessionmaker

AIML_ROOT = Path(__file__).resolve().parents[2] / "aiml"
if str(AIML_ROOT) not in sys.path:
    sys.path.insert(0, str(AIML_ROOT))

from realtime.schemas import MatchAction
from src.persistence.classification import INSUFFICIENT_OBSERVATIONS, SHORT_LIVED

from app.models.event_detection import EventDetection
from app.models.firms_observation import FirmsObservation
from app.models.thermal_event import ThermalEvent
from app.services.event_upsert import (
    process_one_observation,
    process_unassigned_observations,
    refresh_event_persistence,
)
from app.services.observation_identity import compute_observation_hash
from tests.conftest import REQUIRES_POSTGIS

MARKER = "phase4_test_sat"


def _hash_row(**overrides) -> dict:
    row = {
        "latitude": "21.00000",
        "longitude": "86.00000",
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


def _insert_obs(
    session,
    *,
    lat: float,
    lon: float,
    acq_datetime: datetime,
    frp: float = 2.0,
    uniq: str = "0",
) -> FirmsObservation:
    acq_date = acq_datetime.strftime("%Y-%m-%d")
    acq_time = acq_datetime.strftime("%H%M")
    raw = _hash_row(
        latitude=f"{lat:.5f}",
        longitude=f"{lon:.5f}",
        acq_date=acq_date,
        acq_time=acq_time,
        frp=str(frp),
        type=uniq,
    )
    obs_hash = compute_observation_hash(raw)
    obs = FirmsObservation(
        observation_hash=obs_hash,
        latitude=float(lat),
        longitude=float(lon),
        geometry=WKTElement(f"POINT({lon} {lat})", srid=4326),
        acq_date=acq_date,
        acq_time=acq_time,
        acq_datetime=acq_datetime,
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
        type=uniq,
        source_file="phase4_test",
        event_id=None,
    )
    session.add(obs)
    session.flush()
    return obs


def _cleanup(session) -> None:
    """
    Remove Phase 4 test rows without deleting real NRT/historical events.

    A test observation may have *matched* an existing active NRT event; deleting
    that event_id wholesale would CASCADE-wipe real ``event_detections``.
    """
    test_event_ids = list(
        session.execute(
            text(
                "SELECT DISTINCT event_id FROM firms_observations "
                "WHERE satellite = :s AND event_id IS NOT NULL"
            ),
            {"s": MARKER},
        ).scalars()
    )
    session.execute(
        text(
            "DELETE FROM event_detections WHERE observation_hash IN "
            "(SELECT observation_hash FROM firms_observations WHERE satellite = :s)"
        ),
        {"s": MARKER},
    )
    session.execute(
        text("UPDATE firms_observations SET event_id = NULL WHERE satellite = :s"),
        {"s": MARKER},
    )
    for eid in test_event_ids:
        remaining_obs = session.execute(
            text("SELECT COUNT(*) FROM firms_observations WHERE event_id = :e"),
            {"e": eid},
        ).scalar()
        remaining_dets = session.execute(
            text("SELECT COUNT(*) FROM event_detections WHERE event_id = :e"),
            {"e": eid},
        ).scalar()
        if int(remaining_obs or 0) == 0 and int(remaining_dets or 0) == 0:
            session.execute(
                text("DELETE FROM thermal_events WHERE event_id = :e"),
                {"e": eid},
            )
    session.execute(
        text("DELETE FROM firms_observations WHERE satellite = :s"),
        {"s": MARKER},
    )
    session.commit()


@pytest.fixture
def phase4_session(db_engine):
    SessionLocal = sessionmaker(bind=db_engine, autoflush=False, autocommit=False)
    session = SessionLocal()
    _cleanup(session)
    yield session
    _cleanup(session)
    session.close()


@REQUIRES_POSTGIS
def test_one_observation_gets_insufficient_label(phase4_session) -> None:
    db_session = phase4_session
    # Far from India NRT bbox so tests never match live active events.
    t0 = datetime(2026, 9, 5, 6, 0, tzinfo=timezone.utc)
    obs = _insert_obs(db_session, lat=-40.0, lon=-170.0, acq_datetime=t0, uniq="a")
    action = process_one_observation(db_session, obs)
    assert action == MatchAction.CREATED
    event = db_session.scalar(
        select(ThermalEvent).where(ThermalEvent.event_id == obs.event_id)
    )
    assert event is not None
    assert event.detection_count == 1
    assert event.persistence_label == INSUFFICIENT_OBSERVATIONS
    assert event.span_days == 1.0
    assert event.duty_cycle == pytest.approx(1.0)
    assert event.mean_gap_hours is None
    assert event.max_gap_hours is None


@REQUIRES_POSTGIS
def test_three_same_day_short_lived(phase4_session) -> None:
    db_session = phase4_session
    base = datetime(2026, 9, 5, 6, 0, tzinfo=timezone.utc)
    hashes = []
    for i, hours in enumerate([0, 2, 4]):
        obs = _insert_obs(
            db_session,
            lat=-40.01,
            lon=-170.01,
            acq_datetime=base + timedelta(hours=hours),
            uniq=f"s{i}",
        )
        hashes.append(obs.observation_hash)
    stats = process_unassigned_observations(
        db_session, observation_hashes=hashes, commit=False
    )
    assert stats.created == 1
    assert stats.matched == 2
    eid = stats.event_ids_touched[0]
    event = db_session.scalar(select(ThermalEvent).where(ThermalEvent.event_id == eid))
    assert event.detection_count == 3
    assert event.distinct_detection_days == 1
    assert event.persistence_label == SHORT_LIVED


@REQUIRES_POSTGIS
def test_idempotent_refresh(phase4_session) -> None:
    db_session = phase4_session
    t0 = datetime(2026, 9, 5, 8, 0, tzinfo=timezone.utc)
    obs = _insert_obs(db_session, lat=-40.02, lon=-170.02, acq_datetime=t0, uniq="id")
    process_one_observation(db_session, obs)
    event = db_session.scalar(
        select(ThermalEvent).where(ThermalEvent.event_id == obs.event_id)
    )
    snap = (
        event.detection_count,
        event.span_days,
        event.duty_cycle,
        event.persistence_label,
        event.persistence_basis,
    )
    refresh_event_persistence(db_session, obs.event_id)
    db_session.refresh(event)
    snap2 = (
        event.detection_count,
        event.span_days,
        event.duty_cycle,
        event.persistence_label,
        event.persistence_basis,
    )
    assert snap == snap2


@REQUIRES_POSTGIS
def test_adding_detection_updates_only_that_event(phase4_session) -> None:
    db_session = phase4_session
    t0 = datetime(2026, 9, 5, 10, 0, tzinfo=timezone.utc)
    a1 = _insert_obs(db_session, lat=-40.10, lon=-170.10, acq_datetime=t0, uniq="u1")
    b1 = _insert_obs(db_session, lat=-45.50, lon=-175.50, acq_datetime=t0, uniq="u2")
    process_one_observation(db_session, a1)
    process_one_observation(db_session, b1)
    ea = db_session.scalar(select(ThermalEvent).where(ThermalEvent.event_id == a1.event_id))
    eb = db_session.scalar(select(ThermalEvent).where(ThermalEvent.event_id == b1.event_id))
    assert ea.event_id != eb.event_id
    b_before = (
        eb.detection_count,
        eb.persistence_label,
        eb.duty_cycle,
        eb.span_days,
    )

    a2 = _insert_obs(
        db_session,
        lat=-40.1005,
        lon=-170.1005,
        acq_datetime=t0 + timedelta(hours=10),
        uniq="u3",
    )
    process_one_observation(db_session, a2)
    db_session.refresh(ea)
    db_session.refresh(eb)
    assert ea.detection_count == 2
    assert ea.persistence_label == INSUFFICIENT_OBSERVATIONS
    assert (
        eb.detection_count,
        eb.persistence_label,
        eb.duty_cycle,
        eb.span_days,
    ) == b_before


@REQUIRES_POSTGIS
def test_historical_event_untouched(phase4_session) -> None:
    """Insert a fake historical inactive event; NRT processing must not alter it."""
    db_session = phase4_session
    hist = ThermalEvent(
        event_id="EVT_HIST_PHASE4_GUARD",
        event_start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        event_end=datetime(2020, 1, 2, tzinfo=timezone.utc),
        detection_count=99,
        persistence_label="PERSISTENT",
        duty_cycle=0.99,
        span_days=2.0,
        is_active=False,
        centroid_latitude=10.0,
        centroid_longitude=70.0,
        geometry=WKTElement("POINT(70 10)", srid=4326),
    )
    db_session.add(hist)
    db_session.flush()
    before = (
        hist.detection_count,
        hist.persistence_label,
        hist.duty_cycle,
        hist.span_days,
    )
    obs = _insert_obs(
        db_session,
        lat=-40.20,
        lon=-170.20,
        acq_datetime=datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc),
        uniq="h",
    )
    process_one_observation(db_session, obs)
    db_session.refresh(hist)
    assert (
        hist.detection_count,
        hist.persistence_label,
        hist.duty_cycle,
        hist.span_days,
    ) == before
    db_session.execute(
        text("DELETE FROM thermal_events WHERE event_id = 'EVT_HIST_PHASE4_GUARD'")
    )
    db_session.flush()


@REQUIRES_POSTGIS
def test_no_batch_orchestrator_on_db_path(phase4_session) -> None:
    db_session = phase4_session

    def _boom(*_a, **_k):
        raise AssertionError("batch G.1 must not run")

    with patch(
        "src.persistence.persistence_pipeline.run_persistence_characterization",
        side_effect=_boom,
    ):
        obs = _insert_obs(
            db_session,
            lat=-40.30,
            lon=-170.30,
            acq_datetime=datetime(2026, 9, 5, 14, 0, tzinfo=timezone.utc),
            uniq="nb",
        )
        process_one_observation(db_session, obs)
    assert db_session.scalar(
        select(EventDetection.id).where(
            EventDetection.observation_hash == obs.observation_hash
        )
    )
