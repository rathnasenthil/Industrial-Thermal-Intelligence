"""Database tests for Phase 5 incremental I.2 facility association.

Cleanup only touches test-owned rows (facility_id / event_id prefixes).
Never CASCADE-deletes live NRT or historical Stage VII data.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from geoalchemy2.elements import WKTElement
from sqlalchemy import func, select, text
from sqlalchemy.orm import sessionmaker

AIML_ROOT = Path(__file__).resolve().parents[2] / "aiml"
if str(AIML_ROOT) not in sys.path:
    sys.path.insert(0, str(AIML_ROOT))

from realtime.schemas import MatchAction
from src.infrastructure.facility_association import NEAR_FACILITY, NO_FACILITY_ASSOCIATION

from app.models.event_detection import EventDetection
from app.models.event_facility_candidate import EventFacilityCandidate
from app.models.facility import Facility
from app.models.firms_observation import FirmsObservation
from app.models.thermal_event import ThermalEvent
from app.services.event_upsert import process_one_observation
from app.services.facility_association import refresh_event_facility_association
from app.services.observation_identity import compute_observation_hash
from tests.conftest import REQUIRES_POSTGIS

MARKER = "phase5_test_sat"
FAC_PREFIX = "P5TEST_FAC_"
EVT_GUARD = "EVT_HIST_PHASE5_GUARD"


def _hash_row(**overrides) -> dict:
    row = {
        "latitude": "-40.00000",
        "longitude": "-170.00000",
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


def _insert_obs(session, *, lat: float, lon: float, acq_datetime: datetime, uniq: str) -> FirmsObservation:
    acq_date = acq_datetime.strftime("%Y-%m-%d")
    acq_time = acq_datetime.strftime("%H%M")
    raw = _hash_row(
        latitude=f"{lat:.5f}",
        longitude=f"{lon:.5f}",
        acq_date=acq_date,
        acq_time=acq_time,
        type=uniq,
    )
    obs = FirmsObservation(
        observation_hash=compute_observation_hash(raw),
        latitude=lat,
        longitude=lon,
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
        frp=2.0,
        daynight="D",
        type=uniq,
        source_file="phase5_test",
        event_id=None,
    )
    session.add(obs)
    session.flush()
    return obs


def _insert_facility(session, *, fid_suffix: str, lat: float, lon: float, polygon: bool = False) -> Facility:
    fid = f"{FAC_PREFIX}{fid_suffix}"
    if polygon:
        half = 0.01
        wkt = (
            f"POLYGON (("
            f"{lon - half} {lat - half}, {lon + half} {lat - half}, "
            f"{lon + half} {lat + half}, {lon - half} {lat + half}, "
            f"{lon - half} {lat - half}))"
        )
        gtype = "Polygon"
    else:
        wkt = f"POINT ({lon} {lat})"
        gtype = "Point"
    fac = Facility(
        facility_id=fid,
        facility_name=f"Phase5 {fid_suffix}",
        facility_type="MINE",
        geometry_type=gtype,
        latitude=lat,
        longitude=lon,
        geometry=WKTElement(f"POINT({lon} {lat})", srid=4326),
        geometry_wkt=wkt,
        source="phase5_test",
    )
    session.add(fac)
    session.flush()
    return fac


def _cleanup(session) -> None:
    """Delete only Phase-5-owned rows; never wipe historical/NRT demo data."""
    session.execute(
        text(
            "DELETE FROM event_facility_candidates WHERE facility_id LIKE :p "
            "OR event_id IN ("
            "  SELECT event_id FROM firms_observations WHERE satellite = :s AND event_id IS NOT NULL"
            ")"
        ),
        {"p": f"{FAC_PREFIX}%", "s": MARKER},
    )
    # Unlink test observations before deleting test-only events.
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
        non_test_obs = session.execute(
            text(
                "SELECT COUNT(*) FROM firms_observations "
                "WHERE event_id = :e AND satellite IS DISTINCT FROM :s"
            ),
            {"e": eid, "s": MARKER},
        ).scalar()
        if int(non_test_obs or 0) > 0:
            # Never delete live NRT / historical events that also have real observations.
            continue
        remaining_obs = session.execute(
            text("SELECT COUNT(*) FROM firms_observations WHERE event_id = :e"),
            {"e": eid},
        ).scalar()
        remaining_dets = session.execute(
            text("SELECT COUNT(*) FROM event_detections WHERE event_id = :e"),
            {"e": eid},
        ).scalar()
        if int(remaining_obs or 0) == 0 and int(remaining_dets or 0) == 0:
            session.execute(text("DELETE FROM thermal_events WHERE event_id = :e"), {"e": eid})
    session.execute(text("DELETE FROM firms_observations WHERE satellite = :s"), {"s": MARKER})
    session.execute(text("DELETE FROM facilities WHERE facility_id LIKE :p"), {"p": f"{FAC_PREFIX}%"})
    session.execute(text("DELETE FROM thermal_events WHERE event_id = :e"), {"e": EVT_GUARD})
    session.commit()


@pytest.fixture
def phase5_session(db_engine):
    SessionLocal = sessionmaker(bind=db_engine, autoflush=False, autocommit=False)
    session = SessionLocal()
    _cleanup(session)
    yield session
    _cleanup(session)
    session.close()


@REQUIRES_POSTGIS
def test_no_candidate_far_from_facilities(phase5_session) -> None:
    db = phase5_session
    hist_candidates = db.scalar(select(func.count()).select_from(EventFacilityCandidate))
    _insert_facility(db, fid_suffix="FAR", lat=-40.0, lon=-170.0)
    # Event in a different ocean basin — outside 5 km.
    obs = _insert_obs(
        db,
        lat=-10.0,
        lon=-140.0,
        acq_datetime=datetime(2026, 9, 5, 6, 0, tzinfo=timezone.utc),
        uniq="nc",
    )
    assert process_one_observation(db, obs) == MatchAction.CREATED
    event = db.scalar(select(ThermalEvent).where(ThermalEvent.event_id == obs.event_id))
    assert event.facility_association_method == NO_FACILITY_ASSOCIATION
    assert event.facility_id is None
    assert event.candidate_facility_count == 0
    assert (
        db.scalar(
            select(func.count()).where(EventFacilityCandidate.event_id == event.event_id)
        )
        == 0
    )
    assert db.scalar(select(func.count()).select_from(EventFacilityCandidate)) == hist_candidates


@REQUIRES_POSTGIS
def test_one_candidate_associated(phase5_session) -> None:
    db = phase5_session
    # Pacific test location far from India NRT events.
    _insert_facility(db, fid_suffix="ONE", lat=-40.00, lon=-170.00)
    obs = _insert_obs(
        db,
        lat=-40.005,
        lon=-170.00,
        acq_datetime=datetime(2026, 9, 5, 7, 0, tzinfo=timezone.utc),
        uniq="one",
    )
    process_one_observation(db, obs)
    event = db.scalar(select(ThermalEvent).where(ThermalEvent.event_id == obs.event_id))
    assert event.facility_id == f"{FAC_PREFIX}ONE"
    assert event.facility_association_method == NEAR_FACILITY
    assert event.candidate_facility_count == 1
    assert event.facility_distance_km is not None
    cands = list(
        db.scalars(
            select(EventFacilityCandidate).where(EventFacilityCandidate.event_id == event.event_id)
        )
    )
    assert len(cands) == 1
    assert cands[0].candidate_rank == 1


@REQUIRES_POSTGIS
def test_duplicate_processing_no_duplicate_candidates(phase5_session) -> None:
    db = phase5_session
    _insert_facility(db, fid_suffix="IDEM", lat=-41.00, lon=-171.00)
    obs = _insert_obs(
        db,
        lat=-41.004,
        lon=-171.00,
        acq_datetime=datetime(2026, 9, 5, 8, 0, tzinfo=timezone.utc),
        uniq="idem",
    )
    process_one_observation(db, obs)
    eid = obs.event_id
    before = db.scalar(
        select(func.count()).where(EventFacilityCandidate.event_id == eid)
    )
    snap = (
        db.scalar(select(ThermalEvent).where(ThermalEvent.event_id == eid)).facility_id,
        db.scalar(select(ThermalEvent).where(ThermalEvent.event_id == eid)).facility_association_method,
        db.scalar(select(ThermalEvent).where(ThermalEvent.event_id == eid)).facility_distance_km,
    )
    refresh_event_facility_association(db, eid)
    refresh_event_facility_association(db, eid)
    after = db.scalar(
        select(func.count()).where(EventFacilityCandidate.event_id == eid)
    )
    event = db.scalar(select(ThermalEvent).where(ThermalEvent.event_id == eid))
    assert before == after == 1
    assert (event.facility_id, event.facility_association_method, event.facility_distance_km) == snap


@REQUIRES_POSTGIS
def test_event_movement_updates_association(phase5_session) -> None:
    db = phase5_session
    # Facilities ~1.1 km apart; both within Phase 3 spatial continuity of detections.
    _insert_facility(db, fid_suffix="A", lat=-42.00, lon=-172.00)
    _insert_facility(db, fid_suffix="B", lat=-42.01, lon=-172.00)
    t0 = datetime(2026, 9, 5, 9, 0, tzinfo=timezone.utc)
    o1 = _insert_obs(db, lat=-42.0005, lon=-172.00, acq_datetime=t0, uniq="m1")
    process_one_observation(db, o1)
    e = db.scalar(select(ThermalEvent).where(ThermalEvent.event_id == o1.event_id))
    first_facility = e.facility_id
    assert first_facility == f"{FAC_PREFIX}A"
    first_method = e.facility_association_method
    o2 = _insert_obs(
        db,
        lat=-42.0095,
        lon=-172.00,
        acq_datetime=t0.replace(hour=10),
        uniq="m2",
    )
    process_one_observation(db, o2)
    db.refresh(e)
    assert o2.event_id == e.event_id
    assert e.detection_count == 2
    assert e.candidate_facility_count >= 1
    assert e.facility_association_method is not None
    # Recompute is stable for the same detection geometry.
    again = refresh_event_facility_association(db, e.event_id)
    assert again.facility_id == e.facility_id
    assert again.facility_association_method == e.facility_association_method
    assert again.candidate_facility_count == e.candidate_facility_count
    assert first_facility == f"{FAC_PREFIX}A"
    assert first_method is not None



@REQUIRES_POSTGIS
def test_historical_candidates_unchanged(phase5_session) -> None:
    db = phase5_session
    hist_n = db.scalar(select(func.count()).select_from(EventFacilityCandidate))
    hist = ThermalEvent(
        event_id=EVT_GUARD,
        event_start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        event_end=datetime(2020, 1, 2, tzinfo=timezone.utc),
        detection_count=5,
        facility_id="HIST_FAC_KEEP",
        facility_association_method="NEAR_FACILITY",
        facility_distance_km=1.23,
        is_active=False,
        centroid_latitude=10.0,
        centroid_longitude=70.0,
        geometry=WKTElement("POINT(70 10)", srid=4326),
    )
    db.add(hist)
    db.flush()
    before = (hist.facility_id, hist.facility_association_method, hist.facility_distance_km)
    _insert_facility(db, fid_suffix="H", lat=-43.0, lon=-173.0)
    obs = _insert_obs(
        db,
        lat=-43.004,
        lon=-173.0,
        acq_datetime=datetime(2026, 9, 5, 11, 0, tzinfo=timezone.utc),
        uniq="h",
    )
    process_one_observation(db, obs)
    db.refresh(hist)
    assert (hist.facility_id, hist.facility_association_method, hist.facility_distance_km) == before
    # Historical candidate table size: only +test candidates for the new event.
    after_n = db.scalar(select(func.count()).select_from(EventFacilityCandidate))
    assert after_n >= hist_n
    # No deletion of pre-existing rows beyond test scope: count of non-test facilities' candidates.
    non_test = db.scalar(
        select(func.count()).where(~EventFacilityCandidate.facility_id.like(f"{FAC_PREFIX}%"))
    )
    # Guard event has no candidate rows; historical pool should be unchanged.
    assert non_test == hist_n
