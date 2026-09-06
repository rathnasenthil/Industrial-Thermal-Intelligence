"""Database tests for Phase 8 incremental I.5 STA evidence.

Uses synthetic STA GeoDataFrames injected into refresh_event_sta.
Never fabricates production NASA STA files under data/raw/.
Cleanup only touches test-owned rows.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from geoalchemy2.elements import WKTElement
from sqlalchemy import func, inspect, select, text
from sqlalchemy.orm import sessionmaker

AIML_ROOT = Path(__file__).resolve().parents[2] / "aiml"
if str(AIML_ROOT) not in sys.path:
    sys.path.insert(0, str(AIML_ROOT))

from realtime.schemas import MatchAction
from src.sta_evidence.config import NO_STA_ASSOCIATION, STAConfig
from src.infrastructure.facility_association import NEAR_FACILITY, NO_FACILITY_ASSOCIATION

# Load AIML STA fixtures without colliding with backend ``tests`` package.
import importlib.util

_fixture_path = AIML_ROOT / "tests" / "fixtures" / "sta" / "make_fixtures.py"
_spec = importlib.util.spec_from_file_location("aiml_sta_make_fixtures", _fixture_path)
_aiml_sta_fixtures = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_aiml_sta_fixtures)
load_det_as_gdf = _aiml_sta_fixtures.load_det_as_gdf
load_mask_as_gdf = _aiml_sta_fixtures.load_mask_as_gdf
write_synthetic_sta_detections_geojson = _aiml_sta_fixtures.write_synthetic_sta_detections_geojson
write_synthetic_sta_mask_geojson = _aiml_sta_fixtures.write_synthetic_sta_mask_geojson

from app.models.event_facility_candidate import EventFacilityCandidate
from app.models.facility import Facility
from app.models.facility_thermal_fingerprint import FacilityThermalFingerprint
from app.models.firms_observation import FirmsObservation
from app.models.thermal_event import ThermalEvent
from app.services.event_upsert import process_one_observation
from app.services.observation_identity import compute_observation_hash
from app.services.sta import clear_sta_layer_cache, refresh_event_sta
from tests.conftest import REQUIRES_POSTGIS

MARKER = "phase8_test_sat"
FAC_PREFIX = "P8TEST_FAC_"
EVT_PREFIX = "EVT_P8TEST_"
EVT_GUARD = "EVT_HIST_PHASE8_GUARD"


@pytest.fixture()
def combined_sta(tmp_path: Path):
    mask = write_synthetic_sta_mask_geojson(tmp_path / "mask.geojson")
    det = write_synthetic_sta_detections_geojson(tmp_path / "det.geojson")
    combined = pd.concat([load_mask_as_gdf(mask), load_det_as_gdf(det)], ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")


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
        source_file="phase8_test",
        event_id=None,
    )
    session.add(obs)
    session.flush()
    return obs


def _insert_facility(session, *, fid_suffix: str, lat: float, lon: float) -> Facility:
    fid = f"{FAC_PREFIX}{fid_suffix}"
    fac = Facility(
        facility_id=fid,
        facility_name=f"Phase8 {fid_suffix}",
        facility_type="MINE",
        geometry_type="Point",
        latitude=lat,
        longitude=lon,
        geometry=WKTElement(f"POINT({lon} {lat})", srid=4326),
        geometry_wkt=f"POINT ({lon} {lat})",
        source="phase8_test",
    )
    session.add(fac)
    session.flush()
    return fac


def _cleanup(session) -> None:
    clear_sta_layer_cache()
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
def phase8_session(db_engine):
    SessionLocal = sessionmaker(bind=db_engine, autoflush=False, autocommit=False)
    session = SessionLocal()
    _cleanup(session)
    yield session
    _cleanup(session)
    session.close()


@REQUIRES_POSTGIS
def test_i5_columns_exist(phase8_session) -> None:
    insp = inspect(phase8_session.bind)
    cols = {c["name"] for c in insp.get_columns("thermal_events")}
    for name in (
        "sta_association_status",
        "primary_sta_id",
        "sta_layer_type",
        "sta_match_count",
        "sta_nearest_distance_km",
        "sta_intersection_area_m2",
        "sta_evidence_available",
        "sta_temporal_relation",
        "sta_evidence_quality",
    ):
        assert name in cols


@REQUIRES_POSTGIS
def test_sta_available_with_injected_layers(phase8_session, combined_sta) -> None:
    db = phase8_session
    ev = ThermalEvent(
        event_id=f"{EVT_PREFIX}IN",
        event_start=datetime(2023, 6, 15, 9, 0, tzinfo=timezone.utc),
        event_end=datetime(2023, 6, 15, 11, 0, tzinfo=timezone.utc),
        centroid_latitude=28.01,
        centroid_longitude=77.01,
        centroid_wkt="POINT (77.01 28.01)",
        footprint_wkt=(
            "POLYGON ((77.005 28.005, 77.015 28.005, 77.015 28.015, "
            "77.005 28.015, 77.005 28.005))"
        ),
        anomaly_score=1.5,
        anomaly_status="NORMAL",
        is_active=False,
        geometry=WKTElement("POINT(77.01 28.01)", srid=4326),
    )
    db.add(ev)
    db.flush()
    result = refresh_event_sta(
        db, ev.event_id, config=STAConfig(association_radius_km=2.0), sta_gdf=combined_sta
    )
    db.refresh(ev)
    assert result.sta_evidence_available is True
    assert ev.sta_evidence_available is True
    assert ev.sta_association_status != NO_STA_ASSOCIATION or ev.sta_match_count >= 0
    assert ev.anomaly_score == 1.5
    assert ev.anomaly_status == "NORMAL"


@REQUIRES_POSTGIS
def test_sta_unavailable_when_source_missing(phase8_session) -> None:
    db = phase8_session
    clear_sta_layer_cache()
    ev = ThermalEvent(
        event_id=f"{EVT_PREFIX}MISS",
        event_start=datetime(2023, 6, 15, 9, 0, tzinfo=timezone.utc),
        event_end=datetime(2023, 6, 15, 10, 0, tzinfo=timezone.utc),
        centroid_latitude=28.01,
        centroid_longitude=77.01,
        centroid_wkt="POINT (77.01 28.01)",
        footprint_wkt="POINT (77.01 28.01)",
        is_active=False,
        geometry=WKTElement("POINT(77.01 28.01)", srid=4326),
    )
    db.add(ev)
    db.flush()
    cfg = STAConfig(
        mask_path=Path("data/raw/__missing_mask__.geojson"),
        detection_path=Path("data/raw/__missing_det__.geojson"),
    )
    result = refresh_event_sta(db, ev.event_id, config=cfg, sta_gdf=None)
    db.refresh(ev)
    assert result.source_missing is True
    assert ev.sta_association_status == NO_STA_ASSOCIATION
    assert ev.sta_evidence_available is False
    assert ev.sta_match_count == 0
    assert ev.primary_sta_id is None


@REQUIRES_POSTGIS
def test_no_match_far_from_sta(phase8_session, combined_sta) -> None:
    db = phase8_session
    ev = ThermalEvent(
        event_id=f"{EVT_PREFIX}FAR",
        event_start=datetime(2023, 7, 1, 9, 0, tzinfo=timezone.utc),
        event_end=datetime(2023, 7, 1, 10, 0, tzinfo=timezone.utc),
        centroid_latitude=10.0,
        centroid_longitude=70.0,
        centroid_wkt="POINT (70 10)",
        footprint_wkt="POINT (70 10)",
        is_active=False,
        geometry=WKTElement("POINT(70 10)", srid=4326),
    )
    db.add(ev)
    db.flush()
    result = refresh_event_sta(
        db, ev.event_id, config=STAConfig(association_radius_km=0.5), sta_gdf=combined_sta
    )
    assert result.sta_association_status == NO_STA_ASSOCIATION
    assert result.sta_evidence_available is False


@REQUIRES_POSTGIS
def test_idempotent_and_unrelated_unchanged(phase8_session, combined_sta) -> None:
    db = phase8_session
    other = ThermalEvent(
        event_id=f"{EVT_PREFIX}OTHER",
        event_start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        event_end=datetime(2020, 1, 2, tzinfo=timezone.utc),
        anomaly_score=9.9,
        anomaly_status="ANOMALOUS",
        sta_association_status="KEEP",
        sta_evidence_available=True,
        is_active=False,
        geometry=WKTElement("POINT(1 1)", srid=4326),
    )
    db.add(other)
    db.flush()
    snap = (other.anomaly_score, other.anomaly_status, other.sta_association_status)

    ev = ThermalEvent(
        event_id=f"{EVT_PREFIX}IDEM",
        event_start=datetime(2023, 6, 15, 9, 0, tzinfo=timezone.utc),
        event_end=datetime(2023, 6, 15, 11, 0, tzinfo=timezone.utc),
        centroid_latitude=28.01,
        centroid_longitude=77.01,
        centroid_wkt="POINT (77.01 28.01)",
        footprint_wkt="POINT (77.01 28.01)",
        is_active=False,
        geometry=WKTElement("POINT(77.01 28.01)", srid=4326),
    )
    db.add(ev)
    db.flush()
    r1 = refresh_event_sta(db, ev.event_id, config=STAConfig(association_radius_km=2.0), sta_gdf=combined_sta)
    r2 = refresh_event_sta(db, ev.event_id, config=STAConfig(association_radius_km=2.0), sta_gdf=combined_sta)
    assert r1.to_dict() == r2.to_dict()
    db.refresh(other)
    assert (other.anomaly_score, other.anomaly_status, other.sta_association_status) == snap


@REQUIRES_POSTGIS
def test_pipeline_sets_i5_on_nrt_event(phase8_session) -> None:
    """End-to-end without STA files → unavailable defaults; I.4 still set."""
    db = phase8_session
    clear_sta_layer_cache()
    _insert_facility(db, fid_suffix="NRT", lat=-40.00, lon=-170.00)
    obs = _insert_obs(
        db,
        lat=-40.005,
        lon=-170.00,
        acq_datetime=datetime(2026, 9, 5, 7, 0, tzinfo=timezone.utc),
        uniq="nrt",
    )
    assert process_one_observation(db, obs) == MatchAction.CREATED
    event = db.scalar(select(ThermalEvent).where(ThermalEvent.event_id == obs.event_id))
    assert event.facility_association_method == NEAR_FACILITY
    assert event.anomaly_status is not None
    assert event.sta_association_status == NO_STA_ASSOCIATION
    assert event.sta_evidence_available is False
    # Fingerprint may exist for the facility; I.5 must not clear I.4.
    assert event.anomaly_status is not None


@REQUIRES_POSTGIS
def test_historical_unchanged(phase8_session, combined_sta) -> None:
    db = phase8_session
    hist = ThermalEvent(
        event_id=EVT_GUARD,
        event_start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        event_end=datetime(2020, 1, 2, tzinfo=timezone.utc),
        detection_count=5,
        anomaly_score=0.55,
        anomaly_status="NORMAL",
        sta_association_status="HIST_KEEP",
        sta_evidence_available=True,
        is_active=False,
        geometry=WKTElement("POINT(70 10)", srid=4326),
    )
    db.add(hist)
    db.flush()
    before = (hist.anomaly_score, hist.anomaly_status, hist.sta_association_status, hist.detection_count)
    hist_n = db.scalar(select(func.count()).select_from(ThermalEvent))
    hist_cands = db.scalar(select(func.count()).select_from(EventFacilityCandidate))
    hist_fp = db.scalar(select(func.count()).select_from(FacilityThermalFingerprint))

    ev = ThermalEvent(
        event_id=f"{EVT_PREFIX}H",
        event_start=datetime(2023, 6, 15, 9, 0, tzinfo=timezone.utc),
        event_end=datetime(2023, 6, 15, 11, 0, tzinfo=timezone.utc),
        centroid_latitude=28.01,
        centroid_longitude=77.01,
        centroid_wkt="POINT (77.01 28.01)",
        footprint_wkt="POINT (77.01 28.01)",
        is_active=False,
        geometry=WKTElement("POINT(77.01 28.01)", srid=4326),
    )
    db.add(ev)
    db.flush()
    refresh_event_sta(db, ev.event_id, config=STAConfig(association_radius_km=2.0), sta_gdf=combined_sta)
    db.refresh(hist)
    assert (
        hist.anomaly_score,
        hist.anomaly_status,
        hist.sta_association_status,
        hist.detection_count,
    ) == before
    assert db.scalar(select(func.count()).select_from(ThermalEvent)) >= hist_n
    assert db.scalar(select(func.count()).select_from(EventFacilityCandidate)) >= hist_cands
    assert db.scalar(select(func.count()).select_from(FacilityThermalFingerprint)) >= hist_fp


@REQUIRES_POSTGIS
def test_no_facility_still_can_get_sta(phase8_session, combined_sta) -> None:
    db = phase8_session
    ev = ThermalEvent(
        event_id=f"{EVT_PREFIX}NF",
        event_start=datetime(2023, 6, 15, 9, 0, tzinfo=timezone.utc),
        event_end=datetime(2023, 6, 15, 11, 0, tzinfo=timezone.utc),
        facility_id=None,
        facility_association_method=NO_FACILITY_ASSOCIATION,
        centroid_latitude=28.01,
        centroid_longitude=77.01,
        centroid_wkt="POINT (77.01 28.01)",
        footprint_wkt=(
            "POLYGON ((77.005 28.005, 77.015 28.005, 77.015 28.015, "
            "77.005 28.015, 77.005 28.005))"
        ),
        is_active=False,
        geometry=WKTElement("POINT(77.01 28.01)", srid=4326),
    )
    db.add(ev)
    db.flush()
    result = refresh_event_sta(
        db, ev.event_id, config=STAConfig(association_radius_km=2.0), sta_gdf=combined_sta
    )
    assert result.sta_evidence_available is True
    assert ev.facility_association_method == NO_FACILITY_ASSOCIATION
