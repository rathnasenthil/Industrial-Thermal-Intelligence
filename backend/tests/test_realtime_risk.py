"""Database tests for Phase 11 incremental Stage VI risk prioritization."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from geoalchemy2.elements import WKTElement
from sqlalchemy import select, text
from sqlalchemy.orm import sessionmaker

AIML_ROOT = Path(__file__).resolve().parents[2] / "aiml"
if str(AIML_ROOT) not in sys.path:
    sys.path.insert(0, str(AIML_ROOT))

from app.models.facility import Facility
from app.models.firms_observation import FirmsObservation
from app.models.thermal_event import ThermalEvent
from app.services.event_upsert import process_one_observation
from app.services.observation_identity import compute_observation_hash
from app.services.risk import refresh_event_risk
from tests.conftest import REQUIRES_POSTGIS

MARKER = "phase11_test_sat"
FAC_PREFIX = "P11TEST_FAC_"
EVT_PREFIX = "EVT_P11TEST_"
EVT_GUARD = "EVT_HIST_PHASE11_GUARD"


def _hash_row(**overrides) -> dict:
    row = {
        "latitude": "-43.00000",
        "longitude": "-173.00000",
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


def _insert_obs(session, *, lat: float, lon: float, acq_datetime: datetime, uniq: str):
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
        frp=12.0,
        daynight="D",
        type=uniq,
        source_file="phase11_test",
        event_id=None,
    )
    session.add(obs)
    session.flush()
    return obs


def _insert_facility(session, *, fid_suffix: str, lat: float, lon: float) -> Facility:
    fac = Facility(
        facility_id=f"{FAC_PREFIX}{fid_suffix}",
        facility_name=f"Phase11 {fid_suffix}",
        facility_type="POWER_PLANT",
        geometry_type="Point",
        latitude=lat,
        longitude=lon,
        geometry=WKTElement(f"POINT({lon} {lat})", srid=4326),
        geometry_wkt=f"POINT ({lon} {lat})",
        source="phase11_test",
    )
    session.add(fac)
    session.flush()
    return fac


def _cleanup(session) -> None:
    session.execute(
        text("DELETE FROM facility_monthly_thermal_profile WHERE facility_id LIKE :p"),
        {"p": f"{FAC_PREFIX}%"},
    )
    session.execute(
        text("DELETE FROM facility_thermal_fingerprints WHERE facility_id LIKE :p"),
        {"p": f"{FAC_PREFIX}%"},
    )
    session.execute(
        text(
            "DELETE FROM event_facility_candidates WHERE facility_id LIKE :p "
            "OR event_id LIKE :e OR event_id IN ("
            "  SELECT event_id FROM firms_observations WHERE satellite = :s AND event_id IS NOT NULL"
            ")"
        ),
        {"p": f"{FAC_PREFIX}%", "e": f"{EVT_PREFIX}%", "s": MARKER},
    )
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
    session.execute(text("UPDATE firms_observations SET event_id = NULL WHERE satellite = :s"), {"s": MARKER})
    for eid in test_event_ids:
        remaining_obs = session.execute(
            text("SELECT COUNT(*) FROM firms_observations WHERE event_id = :e"), {"e": eid}
        ).scalar()
        remaining_dets = session.execute(
            text("SELECT COUNT(*) FROM event_detections WHERE event_id = :e"), {"e": eid}
        ).scalar()
        if int(remaining_obs or 0) == 0 and int(remaining_dets or 0) == 0:
            session.execute(text("DELETE FROM thermal_events WHERE event_id = :e"), {"e": eid})
    session.execute(text("DELETE FROM firms_observations WHERE satellite = :s"), {"s": MARKER})
    session.execute(text("DELETE FROM thermal_events WHERE event_id LIKE :e"), {"e": f"{EVT_PREFIX}%"})
    session.execute(text("DELETE FROM thermal_events WHERE event_id = :e"), {"e": EVT_GUARD})
    session.execute(text("DELETE FROM facilities WHERE facility_id LIKE :p"), {"p": f"{FAC_PREFIX}%"})
    session.commit()


@pytest.fixture
def phase11_session(db_engine):
    SessionLocal = sessionmaker(bind=db_engine, autoflush=False, autocommit=False)
    session = SessionLocal()
    _cleanup(session)
    yield session
    _cleanup(session)
    session.close()


@REQUIRES_POSTGIS
def test_risk_written_and_historical_unchanged(phase11_session) -> None:
    db = phase11_session
    hist = ThermalEvent(
        event_id=EVT_GUARD,
        event_start=datetime(2023, 1, 1, tzinfo=timezone.utc),
        event_end=datetime(2023, 1, 1, 1, tzinfo=timezone.utc),
        centroid_latitude=-43.0,
        centroid_longitude=-173.0,
        is_active=False,
        detection_count=3,
        risk_score=12.34,
        investigation_priority="LOW",
        source_intelligence_candidate="HIST_GUARD",
        anomaly_status="NORMAL",
        anomaly_score=0.5,
        sta_association_status="NO_STA_ASSOCIATION",
        landcover_available=False,
    )
    db.add(hist)
    db.flush()

    t0 = datetime(2026, 9, 5, 6, 0, tzinfo=timezone.utc)
    _insert_facility(db, fid_suffix="A", lat=-43.0, lon=-173.0)
    obs = _insert_obs(db, lat=-43.0, lon=-173.0, acq_datetime=t0, uniq="r1")
    process_one_observation(db, obs)
    ev = db.scalar(select(ThermalEvent).where(ThermalEvent.event_id == obs.event_id))
    assert ev is not None
    assert ev.risk_score is not None
    assert 0.0 <= float(ev.risk_score) <= 100.0
    assert ev.investigation_priority in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert ev.risk_scoring_version is not None
    fusion_before = (
        ev.source_intelligence_candidate,
        ev.evidence_fusion_score,
        ev.candidate_is_ground_truth,
    )
    i4 = (ev.anomaly_status, ev.anomaly_score)
    i5 = ev.sta_association_status
    i6 = ev.landcover_available
    refresh_event_risk(db, ev.event_id)
    db.refresh(ev)
    db.refresh(hist)
    assert (
        ev.source_intelligence_candidate,
        ev.evidence_fusion_score,
        ev.candidate_is_ground_truth,
    ) == fusion_before
    assert (ev.anomaly_status, ev.anomaly_score) == i4
    assert ev.sta_association_status == i5
    assert ev.landcover_available == i6
    assert hist.risk_score == 12.34
    assert hist.investigation_priority == "LOW"


@REQUIRES_POSTGIS
def test_risk_idempotent(phase11_session) -> None:
    db = phase11_session
    t0 = datetime(2026, 9, 5, 7, 0, tzinfo=timezone.utc)
    _insert_facility(db, fid_suffix="B", lat=-43.1, lon=-173.1)
    obs = _insert_obs(db, lat=-43.1, lon=-173.1, acq_datetime=t0, uniq="r2")
    process_one_observation(db, obs)
    r1 = refresh_event_risk(db, obs.event_id)
    r2 = refresh_event_risk(db, obs.event_id)
    assert r1.to_dict() == r2.to_dict()
