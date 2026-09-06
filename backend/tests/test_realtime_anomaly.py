"""Database tests for Phase 7 incremental I.4 temporal anomaly detection.

Cleanup only touches test-owned rows. Never CASCADE-deletes live NRT /
historical Stage VII data.
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
from src.anomaly_detection.config import INSUFFICIENT_HISTORY, REASON_AMBIGUOUS, REASON_NO_FACILITY
from src.infrastructure.facility_association import AMBIGUOUS, NEAR_FACILITY, NO_FACILITY_ASSOCIATION

from app.models.event_facility_candidate import EventFacilityCandidate
from app.models.facility import Facility
from app.models.facility_monthly_thermal_profile import FacilityMonthlyThermalProfile
from app.models.facility_thermal_fingerprint import FacilityThermalFingerprint
from app.models.firms_observation import FirmsObservation
from app.models.thermal_event import ThermalEvent
from app.services.anomaly import refresh_event_anomaly
from app.services.event_upsert import process_one_observation
from app.services.observation_identity import compute_observation_hash
from tests.conftest import REQUIRES_POSTGIS

MARKER = "phase7_test_sat"
FAC_PREFIX = "P7TEST_FAC_"
EVT_PREFIX = "EVT_P7TEST_"
EVT_GUARD = "EVT_HIST_PHASE7_GUARD"


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
        source_file="phase7_test",
        event_id=None,
    )
    session.add(obs)
    session.flush()
    return obs


def _insert_facility(session, *, fid_suffix: str, lat: float, lon: float) -> Facility:
    fid = f"{FAC_PREFIX}{fid_suffix}"
    fac = Facility(
        facility_id=fid,
        facility_name=f"Phase7 {fid_suffix}",
        facility_type="MINE",
        geometry_type="Point",
        latitude=lat,
        longitude=lon,
        geometry=WKTElement(f"POINT({lon} {lat})", srid=4326),
        geometry_wkt=f"POINT ({lon} {lat})",
        source="phase7_test",
    )
    session.add(fac)
    session.flush()
    return fac


def _insert_confirmed_event(
    session,
    *,
    event_id: str,
    facility_id: str,
    day: int,
    month: int = 1,
    peak_frp: float = 5.0,
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
        peak_frp=peak_frp,
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
def phase7_session(db_engine):
    SessionLocal = sessionmaker(bind=db_engine, autoflush=False, autocommit=False)
    session = SessionLocal()
    _cleanup(session)
    yield session
    _cleanup(session)
    session.close()


@REQUIRES_POSTGIS
def test_confirmed_event_gets_i4_fields(phase7_session) -> None:
    db = phase7_session
    fac = _insert_facility(db, fid_suffix="ONE", lat=-40.00, lon=-170.00)
    for i in range(1, 4):
        _insert_confirmed_event(
            db, event_id=f"{EVT_PREFIX}P{i}", facility_id=fac.facility_id, day=i, peak_frp=5.0
        )
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
    assert event.anomaly_status is not None
    assert event.baseline_observation_count is not None
    assert event.baseline_observation_count >= 3
    assert event.anomaly_unavailable_reason is None
    assert event.anomaly_explanation is not None


@REQUIRES_POSTGIS
def test_insufficient_history(phase7_session) -> None:
    db = phase7_session
    fac = _insert_facility(db, fid_suffix="INS", lat=-40.20, lon=-170.20)
    # No prior confirmed events — first NRT association → insufficient.
    obs = _insert_obs(
        db,
        lat=-40.205,
        lon=-170.20,
        acq_datetime=datetime(2026, 9, 5, 8, 0, tzinfo=timezone.utc),
        uniq="ins",
    )
    process_one_observation(db, obs)
    event = db.scalar(select(ThermalEvent).where(ThermalEvent.event_id == obs.event_id))
    assert event.facility_id == fac.facility_id
    assert event.anomaly_status == INSUFFICIENT_HISTORY
    assert event.anomaly_score is None
    assert event.baseline_observation_count == 0


@REQUIRES_POSTGIS
def test_anomalous_status_with_priors(phase7_session) -> None:
    db = phase7_session
    fac = _insert_facility(db, fid_suffix="ANOM", lat=-40.40, lon=-170.40)
    for i in range(1, 11):
        _insert_confirmed_event(
            db,
            event_id=f"{EVT_PREFIX}A{i}",
            facility_id=fac.facility_id,
            day=min(i, 28),
            month=((i - 1) % 12) + 1,
            peak_frp=5.0,
            detection_count=2,
        )
    # Direct refresh on a synthetic extreme current event.
    cur = _insert_confirmed_event(
        db,
        event_id=f"{EVT_PREFIX}CUR",
        facility_id=fac.facility_id,
        day=20,
        month=6,
        peak_frp=500.0,
        detection_count=200,
    )
    cur.observed_duration_hours = 100.0
    cur.facility_distance_km = 20.0
    db.flush()
    result = refresh_event_anomaly(db, cur.event_id)
    assert result.anomaly_score is not None
    assert result.anomaly_score >= 2.0
    assert result.anomaly_status in ("ELEVATED", "ANOMALOUS")
    db.refresh(cur)
    assert cur.anomaly_status == result.anomaly_status


@REQUIRES_POSTGIS
def test_ambiguous_event(phase7_session) -> None:
    db = phase7_session
    fac_a = _insert_facility(db, fid_suffix="AMB_A", lat=-41.00, lon=-171.00)
    amb = ThermalEvent(
        event_id=f"{EVT_PREFIX}AMB",
        event_start=datetime(2023, 5, 1, tzinfo=timezone.utc),
        event_end=datetime(2023, 5, 1, tzinfo=timezone.utc),
        detection_count=2,
        peak_frp=99.0,
        persistence_label="SHORT_LIVED",
        facility_id=None,
        facility_association_method=AMBIGUOUS,
        facility_attribution_confidence="LOW",
        candidate_facility_count=2,
        is_active=False,
        geometry=WKTElement("POINT(-171 -41)", srid=4326),
    )
    db.add(amb)
    db.flush()
    db.add(
        EventFacilityCandidate(
            event_id=amb.event_id,
            facility_id=fac_a.facility_id,
            spatial_relation="NEAR",
            distance_km=0.1,
            candidate_rank=1,
            candidate_score=0.9,
        )
    )
    db.flush()
    result = refresh_event_anomaly(db, amb.event_id)
    assert result.anomaly_unavailable_reason == REASON_AMBIGUOUS
    assert result.anomaly_status == INSUFFICIENT_HISTORY
    assert result.anomaly_score is None
    assert result.baseline_history_status == "NOT_APPLICABLE"


@REQUIRES_POSTGIS
def test_no_facility_association(phase7_session) -> None:
    db = phase7_session
    _insert_facility(db, fid_suffix="FAR", lat=-40.0, lon=-170.0)
    obs = _insert_obs(
        db,
        lat=-10.0,
        lon=-140.0,
        acq_datetime=datetime(2026, 9, 5, 9, 0, tzinfo=timezone.utc),
        uniq="nofac",
    )
    process_one_observation(db, obs)
    event = db.scalar(select(ThermalEvent).where(ThermalEvent.event_id == obs.event_id))
    assert event.facility_association_method == NO_FACILITY_ASSOCIATION
    assert event.anomaly_unavailable_reason == REASON_NO_FACILITY
    assert event.anomaly_status == INSUFFICIENT_HISTORY
    assert event.anomaly_score is None


@REQUIRES_POSTGIS
def test_unrelated_events_unchanged(phase7_session) -> None:
    db = phase7_session
    fac = _insert_facility(db, fid_suffix="U", lat=-42.00, lon=-172.00)
    other = _insert_confirmed_event(
        db, event_id=f"{EVT_PREFIX}OTHER", facility_id=fac.facility_id, day=1
    )
    other.anomaly_score = 1.23
    other.anomaly_status = "NORMAL"
    other.anomaly_explanation = "KEEP"
    db.flush()
    snap = (other.anomaly_score, other.anomaly_status, other.anomaly_explanation)

    fac2 = _insert_facility(db, fid_suffix="U2", lat=-42.50, lon=-172.50)
    obs = _insert_obs(
        db,
        lat=-42.504,
        lon=-172.50,
        acq_datetime=datetime(2026, 9, 5, 10, 0, tzinfo=timezone.utc),
        uniq="unrel",
    )
    process_one_observation(db, obs)
    db.refresh(other)
    assert (other.anomaly_score, other.anomaly_status, other.anomaly_explanation) == snap
    assert fac2.facility_id is not None


@REQUIRES_POSTGIS
def test_current_not_self_scored(phase7_session) -> None:
    db = phase7_session
    fac = _insert_facility(db, fid_suffix="SELF", lat=-43.00, lon=-173.00)
    for i in range(1, 4):
        _insert_confirmed_event(
            db, event_id=f"{EVT_PREFIX}S{i}", facility_id=fac.facility_id, day=i, peak_frp=5.0
        )
    cur = _insert_confirmed_event(
        db,
        event_id=f"{EVT_PREFIX}SCUR",
        facility_id=fac.facility_id,
        day=10,
        peak_frp=100.0,
    )
    result = refresh_event_anomaly(db, cur.event_id)
    assert result.baseline_observation_count == 3
    # Constant prior FRP=5, current=100 → constant_mismatch deviation 3.0
    assert result.peak_frp_deviation == pytest.approx(3.0)


@REQUIRES_POSTGIS
def test_association_change_a_to_b_scores_against_b(phase7_session) -> None:
    db = phase7_session
    fac_a = _insert_facility(db, fid_suffix="MOVE_A", lat=-44.00, lon=-174.00)
    fac_b = _insert_facility(db, fid_suffix="MOVE_B", lat=-44.50, lon=-174.50)
    for i in range(1, 4):
        _insert_confirmed_event(
            db, event_id=f"{EVT_PREFIX}BA{i}", facility_id=fac_b.facility_id, day=i, peak_frp=5.0
        )
    ev = _insert_confirmed_event(
        db, event_id=f"{EVT_PREFIX}MOVE", facility_id=fac_a.facility_id, day=5, peak_frp=5.0
    )
    refresh_event_anomaly(db, ev.event_id)
    assert ev.baseline_observation_count == 0  # alone on A

    ev.facility_id = fac_b.facility_id
    db.flush()
    result = refresh_event_anomaly(db, ev.event_id)
    assert result.baseline_observation_count == 3  # B's three priors
    assert result.anomaly_unavailable_reason is None


@REQUIRES_POSTGIS
def test_idempotent_refresh(phase7_session) -> None:
    db = phase7_session
    fac = _insert_facility(db, fid_suffix="IDEM", lat=-45.00, lon=-175.00)
    for i in range(1, 4):
        _insert_confirmed_event(
            db, event_id=f"{EVT_PREFIX}I{i}", facility_id=fac.facility_id, day=i
        )
    obs = _insert_obs(
        db,
        lat=-45.004,
        lon=-175.00,
        acq_datetime=datetime(2026, 9, 5, 11, 0, tzinfo=timezone.utc),
        uniq="idem",
    )
    process_one_observation(db, obs)
    eid = obs.event_id
    e1 = db.scalar(select(ThermalEvent).where(ThermalEvent.event_id == eid))
    snap = (
        e1.anomaly_score,
        e1.anomaly_status,
        e1.baseline_observation_count,
        e1.peak_frp_deviation,
        e1.anomaly_explanation,
    )
    refresh_event_anomaly(db, eid)
    refresh_event_anomaly(db, eid)
    e2 = db.scalar(select(ThermalEvent).where(ThermalEvent.event_id == eid))
    assert (
        e2.anomaly_score,
        e2.anomaly_status,
        e2.baseline_observation_count,
        e2.peak_frp_deviation,
        e2.anomaly_explanation,
    ) == snap


@REQUIRES_POSTGIS
def test_historical_events_unchanged(phase7_session) -> None:
    db = phase7_session
    hist = ThermalEvent(
        event_id=EVT_GUARD,
        event_start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        event_end=datetime(2020, 1, 2, tzinfo=timezone.utc),
        detection_count=5,
        facility_id="HIST_FAC_KEEP",
        facility_association_method="NEAR_FACILITY",
        anomaly_score=0.55,
        anomaly_status="NORMAL",
        is_active=False,
        geometry=WKTElement("POINT(70 10)", srid=4326),
    )
    db.add(hist)
    db.flush()
    before = (hist.anomaly_score, hist.anomaly_status, hist.detection_count)
    hist_events = db.scalar(select(func.count()).select_from(ThermalEvent))
    hist_fac = db.scalar(select(func.count()).select_from(Facility))
    hist_cands = db.scalar(select(func.count()).select_from(EventFacilityCandidate))

    _insert_facility(db, fid_suffix="H", lat=-46.0, lon=-176.0)
    obs = _insert_obs(
        db,
        lat=-46.004,
        lon=-176.0,
        acq_datetime=datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc),
        uniq="h",
    )
    process_one_observation(db, obs)
    db.refresh(hist)
    assert (hist.anomaly_score, hist.anomaly_status, hist.detection_count) == before
    assert db.scalar(select(func.count()).select_from(ThermalEvent)) >= hist_events
    assert db.scalar(select(func.count()).select_from(Facility)) >= hist_fac
    assert db.scalar(select(func.count()).select_from(EventFacilityCandidate)) >= hist_cands


@REQUIRES_POSTGIS
def test_i4_does_not_query_fingerprint_tables(phase7_session) -> None:
    db = phase7_session
    fac = _insert_facility(db, fid_suffix="FP", lat=-47.00, lon=-177.00)
    for i in range(1, 4):
        _insert_confirmed_event(
            db, event_id=f"{EVT_PREFIX}F{i}", facility_id=fac.facility_id, day=i
        )
    cur = _insert_confirmed_event(
        db, event_id=f"{EVT_PREFIX}FCUR", facility_id=fac.facility_id, day=10
    )
    fp_before = db.scalar(select(func.count()).select_from(FacilityThermalFingerprint))
    monthly_before = db.scalar(select(func.count()).select_from(FacilityMonthlyThermalProfile))
    # Module must not import fingerprint models for scoring.
    import app.services.anomaly as anomaly_mod

    assert not hasattr(anomaly_mod, "FacilityThermalFingerprint")
    assert not hasattr(anomaly_mod, "FacilityMonthlyThermalProfile")
    refresh_event_anomaly(db, cur.event_id)
    assert db.scalar(select(func.count()).select_from(FacilityThermalFingerprint)) == fp_before
    assert (
        db.scalar(select(func.count()).select_from(FacilityMonthlyThermalProfile))
        == monthly_before
    )

@REQUIRES_POSTGIS
def test_non_test_counts_stable(phase7_session) -> None:
    db = phase7_session
    non_test_fp = db.scalar(
        select(func.count()).where(~FacilityThermalFingerprint.facility_id.like(f"{FAC_PREFIX}%"))
    )
    non_test_cands = db.scalar(
        select(func.count()).where(~EventFacilityCandidate.facility_id.like(f"{FAC_PREFIX}%"))
    )
    _insert_facility(db, fid_suffix="CNT", lat=-48.0, lon=-178.0)
    obs = _insert_obs(
        db,
        lat=-48.004,
        lon=-178.0,
        acq_datetime=datetime(2026, 9, 5, 13, 0, tzinfo=timezone.utc),
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
            select(func.count()).where(~EventFacilityCandidate.facility_id.like(f"{FAC_PREFIX}%"))
        )
        == non_test_cands
    )
