"""Database tests for Phase 6 incremental I.3 facility fingerprinting.

Cleanup only touches test-owned rows (facility_id / satellite markers).
Never CASCADE-deletes live NRT or historical Stage VII data.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from geoalchemy2.elements import WKTElement
from sqlalchemy import func, inspect, select, text
from sqlalchemy.orm import sessionmaker

AIML_ROOT = Path(__file__).resolve().parents[2] / "aiml"
if str(AIML_ROOT) not in sys.path:
    sys.path.insert(0, str(AIML_ROOT))

from realtime.schemas import MatchAction
from src.infrastructure.facility_association import AMBIGUOUS, NEAR_FACILITY

from app.models.event_detection import EventDetection
from app.models.event_facility_candidate import EventFacilityCandidate
from app.models.facility import Facility
from app.models.facility_monthly_thermal_profile import FacilityMonthlyThermalProfile
from app.models.facility_thermal_fingerprint import FacilityThermalFingerprint
from app.models.firms_observation import FirmsObservation
from app.models.thermal_event import ThermalEvent
from app.services.event_upsert import process_one_observation
from app.services.facility_fingerprint import (
    refresh_facility_fingerprint,
    refresh_fingerprints_for_event,
)
from app.services.observation_identity import compute_observation_hash
from tests.conftest import REQUIRES_POSTGIS

MARKER = "phase6_test_sat"
FAC_PREFIX = "P6TEST_FAC_"
EVT_GUARD = "EVT_HIST_PHASE6_GUARD"
EVT_PREFIX = "EVT_P6TEST_"


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
        source_file="phase6_test",
        event_id=None,
    )
    session.add(obs)
    session.flush()
    return obs


def _insert_facility(session, *, fid_suffix: str, lat: float, lon: float) -> Facility:
    fid = f"{FAC_PREFIX}{fid_suffix}"
    wkt = f"POINT ({lon} {lat})"
    fac = Facility(
        facility_id=fid,
        facility_name=f"Phase6 {fid_suffix}",
        facility_type="MINE",
        geometry_type="Point",
        latitude=lat,
        longitude=lon,
        geometry=WKTElement(f"POINT({lon} {lat})", srid=4326),
        geometry_wkt=wkt,
        source="phase6_test",
    )
    session.add(fac)
    session.flush()
    return fac


def _insert_confirmed_event(
    session,
    *,
    event_id: str,
    facility_id: str,
    month: int,
    day: int = 1,
    detection_count: int = 2,
) -> ThermalEvent:
    start = datetime(2023, month, day, 6, 0, tzinfo=timezone.utc)
    ev = ThermalEvent(
        event_id=event_id,
        event_start=start,
        event_end=start,
        detection_count=detection_count,
        distinct_detection_days=1,
        observed_duration_hours=1.0,
        day_detection_count=detection_count,
        night_detection_count=0,
        peak_frp=5.0,
        persistence_label="SHORT_LIVED",
        facility_id=facility_id,
        facility_association_method=NEAR_FACILITY,
        facility_attribution_confidence="MEDIUM",
        facility_distance_km=1.0,
        candidate_facility_count=1,
        is_active=False,
        centroid_latitude=-40.0,
        centroid_longitude=-170.0,
        geometry=WKTElement("POINT(-170 -40)", srid=4326),
    )
    session.add(ev)
    session.flush()
    return ev


def _cleanup(session) -> None:
    """Delete only Phase-6-owned rows; never wipe historical/NRT demo data."""
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
    session.execute(text("DELETE FROM event_facility_candidates WHERE event_id LIKE :e"), {"e": f"{EVT_PREFIX}%"})
    session.execute(text("DELETE FROM thermal_events WHERE event_id LIKE :e"), {"e": f"{EVT_PREFIX}%"})
    session.execute(text("DELETE FROM thermal_events WHERE event_id = :e"), {"e": EVT_GUARD})
    session.execute(text("DELETE FROM facilities WHERE facility_id LIKE :p"), {"p": f"{FAC_PREFIX}%"})
    session.commit()


@pytest.fixture
def phase6_session(db_engine):
    SessionLocal = sessionmaker(bind=db_engine, autoflush=False, autocommit=False)
    session = SessionLocal()
    _cleanup(session)
    yield session
    _cleanup(session)
    session.close()


@REQUIRES_POSTGIS
def test_fingerprint_table_exists(phase6_session) -> None:
    insp = inspect(phase6_session.bind)
    assert "facility_thermal_fingerprints" in insp.get_table_names()
    cols = {c["name"] for c in insp.get_columns("facility_thermal_fingerprints")}
    assert "facility_id" in cols
    assert "fingerprint_status" in cols
    assert "event_count" in cols


@REQUIRES_POSTGIS
def test_monthly_profile_table_exists(phase6_session) -> None:
    insp = inspect(phase6_session.bind)
    assert "facility_monthly_thermal_profile" in insp.get_table_names()
    cols = {c["name"] for c in insp.get_columns("facility_monthly_thermal_profile")}
    assert {"facility_id", "month", "event_count", "detection_count", "event_fraction"} <= cols


@REQUIRES_POSTGIS
def test_confirmed_event_updates_one_facility(phase6_session) -> None:
    db = phase6_session
    fac = _insert_facility(db, fid_suffix="ONE", lat=-40.00, lon=-170.00)
    unrelated = _insert_facility(db, fid_suffix="OTHER", lat=-45.00, lon=-175.00)
    refresh_facility_fingerprint(db, unrelated.facility_id)
    before_other = db.scalar(
        select(FacilityThermalFingerprint).where(
            FacilityThermalFingerprint.facility_id == unrelated.facility_id
        )
    )
    assert before_other.event_count == 0
    other_updated = before_other.updated_at

    obs = _insert_obs(
        db,
        lat=-40.005,
        lon=-170.00,
        acq_datetime=datetime(2026, 9, 5, 7, 0, tzinfo=timezone.utc),
        uniq="one",
    )
    assert process_one_observation(db, obs) == MatchAction.CREATED
    event = db.scalar(select(ThermalEvent).where(ThermalEvent.event_id == obs.event_id))
    assert event.facility_id == fac.facility_id
    assert event.facility_association_method == NEAR_FACILITY

    fp = db.scalar(
        select(FacilityThermalFingerprint).where(
            FacilityThermalFingerprint.facility_id == fac.facility_id
        )
    )
    assert fp is not None
    assert fp.event_count == 1
    assert fp.fingerprint_observation_count == 1
    assert fp.fingerprint_status == "INSUFFICIENT_HISTORY"

    db.refresh(before_other)
    assert before_other.event_count == 0
    assert before_other.updated_at == other_updated


@REQUIRES_POSTGIS
def test_unrelated_facilities_unchanged(phase6_session) -> None:
    db = phase6_session
    fac_a = _insert_facility(db, fid_suffix="A", lat=-40.10, lon=-170.10)
    fac_b = _insert_facility(db, fid_suffix="B", lat=-40.50, lon=-170.50)
    _insert_confirmed_event(db, event_id=f"{EVT_PREFIX}B1", facility_id=fac_b.facility_id, month=2)
    refresh_facility_fingerprint(db, fac_b.facility_id)
    snap = db.scalar(
        select(FacilityThermalFingerprint).where(
            FacilityThermalFingerprint.facility_id == fac_b.facility_id
        )
    )
    snap_vals = (snap.event_count, snap.detection_count, snap.fingerprint_status, snap.updated_at)

    obs = _insert_obs(
        db,
        lat=-40.105,
        lon=-170.10,
        acq_datetime=datetime(2026, 9, 5, 8, 0, tzinfo=timezone.utc),
        uniq="unrel",
    )
    process_one_observation(db, obs)
    db.refresh(snap)
    assert (snap.event_count, snap.detection_count, snap.fingerprint_status, snap.updated_at) == snap_vals
    assert (
        db.scalar(
            select(FacilityThermalFingerprint.event_count).where(
                FacilityThermalFingerprint.facility_id == fac_a.facility_id
            )
        )
        == 1
    )


@REQUIRES_POSTGIS
def test_ambiguous_does_not_increase_event_count(phase6_session) -> None:
    db = phase6_session
    fac_a = _insert_facility(db, fid_suffix="AMB_A", lat=-41.00, lon=-171.00)
    fac_b = _insert_facility(db, fid_suffix="AMB_B", lat=-41.002, lon=-171.00)
    # Seed empty fingerprints
    refresh_facility_fingerprint(db, fac_a.facility_id)
    refresh_facility_fingerprint(db, fac_b.facility_id)

    amb_id = f"{EVT_PREFIX}AMB"
    amb = ThermalEvent(
        event_id=amb_id,
        event_start=datetime(2023, 5, 1, tzinfo=timezone.utc),
        event_end=datetime(2023, 5, 1, tzinfo=timezone.utc),
        detection_count=2,
        distinct_detection_days=1,
        observed_duration_hours=1.0,
        day_detection_count=2,
        night_detection_count=0,
        peak_frp=3.0,
        persistence_label="SHORT_LIVED",
        facility_id=None,
        facility_association_method=AMBIGUOUS,
        facility_attribution_confidence="LOW",
        facility_distance_km=None,
        candidate_facility_count=2,
        is_active=False,
        centroid_latitude=-41.001,
        centroid_longitude=-171.0,
        geometry=WKTElement("POINT(-171 -41.001)", srid=4326),
    )
    db.add(amb)
    db.flush()
    db.add(
        EventFacilityCandidate(
            event_id=amb_id,
            facility_id=fac_a.facility_id,
            spatial_relation="NEAR",
            distance_km=0.1,
            candidate_rank=1,
            candidate_score=0.9,
        )
    )
    db.add(
        EventFacilityCandidate(
            event_id=amb_id,
            facility_id=fac_b.facility_id,
            spatial_relation="NEAR",
            distance_km=0.15,
            candidate_rank=2,
            candidate_score=0.85,
        )
    )
    db.flush()

    stats = refresh_fingerprints_for_event(db, amb_id)
    assert set(stats.facility_ids) == {fac_a.facility_id, fac_b.facility_id}
    for fid in (fac_a.facility_id, fac_b.facility_id):
        fp = db.scalar(
            select(FacilityThermalFingerprint).where(FacilityThermalFingerprint.facility_id == fid)
        )
        assert fp.event_count == 0
        assert fp.fingerprint_status == "NO_OBSERVATIONS"
        assert fp.ambiguous_candidate_opportunity_count == 1
        monthly_n = db.scalar(
            select(func.count()).where(FacilityMonthlyThermalProfile.facility_id == fid)
        )
        assert monthly_n == 0


@REQUIRES_POSTGIS
def test_idempotent_refresh(phase6_session) -> None:
    db = phase6_session
    fac = _insert_facility(db, fid_suffix="IDEM", lat=-42.00, lon=-172.00)
    obs = _insert_obs(
        db,
        lat=-42.004,
        lon=-172.00,
        acq_datetime=datetime(2026, 9, 5, 9, 0, tzinfo=timezone.utc),
        uniq="idem",
    )
    process_one_observation(db, obs)
    eid = obs.event_id
    fp1 = db.scalar(
        select(FacilityThermalFingerprint).where(
            FacilityThermalFingerprint.facility_id == fac.facility_id
        )
    )
    snap = (
        fp1.event_count,
        fp1.detection_count,
        fp1.fingerprint_status,
        fp1.peak_frp_median,
    )
    monthly1 = db.scalar(
        select(func.count()).where(FacilityMonthlyThermalProfile.facility_id == fac.facility_id)
    )
    refresh_fingerprints_for_event(db, eid, previous_facility_id=fac.facility_id)
    refresh_fingerprints_for_event(db, eid, previous_facility_id=fac.facility_id)
    fp2 = db.scalar(
        select(FacilityThermalFingerprint).where(
            FacilityThermalFingerprint.facility_id == fac.facility_id
        )
    )
    monthly2 = db.scalar(
        select(func.count()).where(FacilityMonthlyThermalProfile.facility_id == fac.facility_id)
    )
    assert (
        fp2.event_count,
        fp2.detection_count,
        fp2.fingerprint_status,
        fp2.peak_frp_median,
    ) == snap
    assert monthly1 == monthly2 == 1


@REQUIRES_POSTGIS
def test_association_change_a_to_b_refreshes_both(phase6_session) -> None:
    db = phase6_session
    fac_a = _insert_facility(db, fid_suffix="MOVE_A", lat=-43.00, lon=-173.00)
    fac_b = _insert_facility(db, fid_suffix="MOVE_B", lat=-43.50, lon=-173.50)
    ev = _insert_confirmed_event(
        db, event_id=f"{EVT_PREFIX}MOVE", facility_id=fac_a.facility_id, month=4
    )
    refresh_facility_fingerprint(db, fac_a.facility_id)
    refresh_facility_fingerprint(db, fac_b.facility_id)
    assert (
        db.scalar(
            select(FacilityThermalFingerprint.event_count).where(
                FacilityThermalFingerprint.facility_id == fac_a.facility_id
            )
        )
        == 1
    )
    assert (
        db.scalar(
            select(FacilityThermalFingerprint.event_count).where(
                FacilityThermalFingerprint.facility_id == fac_b.facility_id
            )
        )
        == 0
    )

    # Move association A → B
    ev.facility_id = fac_b.facility_id
    db.flush()
    stats = refresh_fingerprints_for_event(
        db, ev.event_id, previous_facility_id=fac_a.facility_id
    )
    assert set(stats.facility_ids) == {fac_a.facility_id, fac_b.facility_id}
    assert (
        db.scalar(
            select(FacilityThermalFingerprint.event_count).where(
                FacilityThermalFingerprint.facility_id == fac_a.facility_id
            )
        )
        == 0
    )
    assert (
        db.scalar(
            select(FacilityThermalFingerprint.event_count).where(
                FacilityThermalFingerprint.facility_id == fac_b.facility_id
            )
        )
        == 1
    )


@REQUIRES_POSTGIS
def test_historical_events_remain_unchanged(phase6_session) -> None:
    db = phase6_session
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
        anomaly_score=0.42,
    )
    db.add(hist)
    db.flush()
    before = (
        hist.facility_id,
        hist.facility_association_method,
        hist.facility_distance_km,
        hist.anomaly_score,
        hist.detection_count,
    )
    hist_events = db.scalar(select(func.count()).select_from(ThermalEvent))
    hist_facilities = db.scalar(select(func.count()).select_from(Facility))
    hist_cands = db.scalar(select(func.count()).select_from(EventFacilityCandidate))

    _insert_facility(db, fid_suffix="H", lat=-44.0, lon=-174.0)
    obs = _insert_obs(
        db,
        lat=-44.004,
        lon=-174.0,
        acq_datetime=datetime(2026, 9, 5, 11, 0, tzinfo=timezone.utc),
        uniq="h",
    )
    process_one_observation(db, obs)
    db.refresh(hist)
    assert (
        hist.facility_id,
        hist.facility_association_method,
        hist.facility_distance_km,
        hist.anomaly_score,
        hist.detection_count,
    ) == before
    # Counts only grow by test-owned rows, never shrink historical pool.
    assert db.scalar(select(func.count()).select_from(ThermalEvent)) >= hist_events
    assert db.scalar(select(func.count()).select_from(Facility)) >= hist_facilities
    assert db.scalar(select(func.count()).select_from(EventFacilityCandidate)) >= hist_cands


@REQUIRES_POSTGIS
def test_existing_counts_remain_stable_for_non_test_rows(phase6_session) -> None:
    db = phase6_session
    non_test_fp = db.scalar(
        select(func.count()).where(~FacilityThermalFingerprint.facility_id.like(f"{FAC_PREFIX}%"))
    )
    non_test_monthly = db.scalar(
        select(func.count()).where(~FacilityMonthlyThermalProfile.facility_id.like(f"{FAC_PREFIX}%"))
    )
    non_test_cands = db.scalar(
        select(func.count()).where(~EventFacilityCandidate.facility_id.like(f"{FAC_PREFIX}%"))
    )
    _insert_facility(db, fid_suffix="CNT", lat=-45.0, lon=-175.0)
    obs = _insert_obs(
        db,
        lat=-45.004,
        lon=-175.0,
        acq_datetime=datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc),
        uniq="cnt",
    )
    process_one_observation(db, obs)
    assert (
        db.scalar(
            select(func.count()).where(
                ~FacilityThermalFingerprint.facility_id.like(f"{FAC_PREFIX}%")
            )
        )
        == non_test_fp
    )
    assert (
        db.scalar(
            select(func.count()).where(
                ~FacilityMonthlyThermalProfile.facility_id.like(f"{FAC_PREFIX}%")
            )
        )
        == non_test_monthly
    )
    assert (
        db.scalar(
            select(func.count()).where(~EventFacilityCandidate.facility_id.like(f"{FAC_PREFIX}%"))
        )
        == non_test_cands
    )
