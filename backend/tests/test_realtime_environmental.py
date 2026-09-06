"""Database tests for Phase 9 incremental I.6 environmental context.

Uses synthetic fixtures under tmp_path. Never fabricates production datasets
under aiml/data/external/. Cleanup only touches test-owned rows.
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

import importlib.util

_fixture_path = AIML_ROOT / "tests" / "fixtures" / "environmental_context" / "make_fixtures.py"
_spec = importlib.util.spec_from_file_location("aiml_env_make_fixtures", _fixture_path)
_aiml_env_fixtures = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_aiml_env_fixtures)
write_water_geojson = _aiml_env_fixtures.write_water_geojson

from src.environmental_context.config import EnvironmentalContextConfig

from app.models.facility import Facility
from app.models.facility_thermal_fingerprint import FacilityThermalFingerprint
from app.models.firms_observation import FirmsObservation
from app.models.thermal_event import ThermalEvent
from app.services.environmental import refresh_event_environmental
from app.services.event_upsert import process_one_observation
from app.services.observation_identity import compute_observation_hash
from tests.conftest import REQUIRES_POSTGIS

MARKER = "phase9_test_sat"
FAC_PREFIX = "P9TEST_FAC_"
EVT_PREFIX = "EVT_P9TEST_"
EVT_GUARD = "EVT_HIST_PHASE9_GUARD"


def _missing_cfg(**overrides) -> EnvironmentalContextConfig:
    base = dict(
        landcover_raster_path=Path("data/external/does_not_exist_lc.tif"),
        landcover_vector_path=Path("data/external/does_not_exist_lc.geojson"),
        vegetation_path=Path("data/external/does_not_exist_veg.geojson"),
        builtup_path=Path("data/external/does_not_exist_built.geojson"),
        water_path=Path("data/external/does_not_exist_water.geojson"),
        agriculture_path=Path("data/external/does_not_exist_ag.geojson"),
        satellite_raster_path=Path("data/external/does_not_exist_sat.tif"),
    )
    base.update(overrides)
    return EnvironmentalContextConfig(**base)


def _hash_row(**overrides) -> dict:
    row = {
        "latitude": "-41.00000",
        "longitude": "-171.00000",
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
        source_file="phase9_test",
        event_id=None,
    )
    session.add(obs)
    session.flush()
    return obs


def _insert_facility(session, *, fid_suffix: str, lat: float, lon: float) -> Facility:
    fid = f"{FAC_PREFIX}{fid_suffix}"
    fac = Facility(
        facility_id=fid,
        facility_name=f"Phase9 {fid_suffix}",
        facility_type="MINE",
        geometry_type="Point",
        latitude=lat,
        longitude=lon,
        geometry=WKTElement(f"POINT({lon} {lat})", srid=4326),
        geometry_wkt=f"POINT ({lon} {lat})",
        source="phase9_test",
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
def phase9_session(db_engine):
    SessionLocal = sessionmaker(bind=db_engine, autoflush=False, autocommit=False)
    session = SessionLocal()
    _cleanup(session)
    yield session
    _cleanup(session)
    session.close()


@REQUIRES_POSTGIS
def test_i6_columns_exist(phase9_session) -> None:
    insp = inspect(phase9_session.bind)
    cols = {c["name"] for c in insp.get_columns("thermal_events")}
    for name in (
        "landcover_available",
        "landcover_source",
        "dominant_landcover_class",
        "vegetation_present",
        "builtup_present",
        "water_present",
        "agriculture_present",
        "satellite_value",
        "satellite_value_name",
    ):
        assert name in cols


@REQUIRES_POSTGIS
def test_source_unavailable_writes_false_flags(phase9_session) -> None:
    db = phase9_session
    t0 = datetime(2026, 9, 5, 6, 0, tzinfo=timezone.utc)
    fac = _insert_facility(db, fid_suffix="A", lat=-41.0, lon=-171.0)
    obs = _insert_obs(db, lat=-41.0, lon=-171.0, acq_datetime=t0, uniq="u1")
    process_one_observation(db, obs)
    assert obs.event_id is not None
    ev = db.scalar(select(ThermalEvent).where(ThermalEvent.event_id == obs.event_id))
    assert ev is not None
    # Default production paths under aiml/data/external/ are empty → unavailable.
    assert ev.landcover_available is False
    assert ev.vegetation_context_available is False
    assert ev.builtup_context_available is False
    assert ev.water_context_available is False
    assert ev.agriculture_context_available is False
    assert ev.satellite_context_available is False
    assert ev.dominant_landcover_class is None
    assert ev.dominant_landcover_fraction is None
    assert ev.water_present is None
    assert ev.satellite_value is None
    # Facility association still ran; fingerprint may exist for NEAR_FACILITY
    assert fac.facility_id


@REQUIRES_POSTGIS
def test_water_fixture_updates_only_current_event(phase9_session, tmp_path: Path) -> None:
    db = phase9_session
    water = write_water_geojson(tmp_path / "water.geojson")
    cfg = _missing_cfg(water_path=water, context_buffer_km=1.0, broad_context_buffer_km=5.0)

    hist = ThermalEvent(
        event_id=EVT_GUARD,
        event_start=datetime(2023, 1, 1, tzinfo=timezone.utc),
        event_end=datetime(2023, 1, 1, 1, tzinfo=timezone.utc),
        centroid_latitude=28.01,
        centroid_longitude=77.01,
        centroid_wkt="POINT (77.01 28.01)",
        footprint_wkt="POINT (77.01 28.01)",
        is_active=False,
        detection_count=3,
        landcover_available=True,
        dominant_landcover_class="HIST_GUARD",
        water_context_available=True,
        water_present=True,
        anomaly_status="NORMAL",
        anomaly_score=1.5,
        sta_association_status="NO_STA_ASSOCIATION",
        sta_evidence_available=False,
    )
    db.add(hist)
    db.flush()

    # Place current event at lake centroid
    t0 = datetime(2026, 9, 5, 6, 0, tzinfo=timezone.utc)
    _insert_facility(db, fid_suffix="W", lat=28.01, lon=77.01)
    obs = _insert_obs(db, lat=28.01, lon=77.01, acq_datetime=t0, uniq="w1")
    process_one_observation(db, obs)
    cur = db.scalar(select(ThermalEvent).where(ThermalEvent.event_id == obs.event_id))
    assert cur is not None

    # Force I.6 with fixture (pipeline already wrote unavailable via missing files)
    i4_before = (cur.anomaly_status, cur.anomaly_score)
    i5_before = (cur.sta_association_status, cur.sta_evidence_available, cur.sta_match_count)
    refresh_event_environmental(db, cur.event_id, config=cfg)
    db.refresh(cur)
    db.refresh(hist)

    assert cur.water_context_available is True
    assert cur.water_present is True
    assert hist.dominant_landcover_class == "HIST_GUARD"
    assert hist.water_present is True
    assert (cur.anomaly_status, cur.anomaly_score) == i4_before
    assert (
        cur.sta_association_status,
        cur.sta_evidence_available,
        cur.sta_match_count,
    ) == i5_before


@REQUIRES_POSTGIS
def test_landcover_fixture_and_idempotency(phase9_session, tmp_path: Path) -> None:
    """Use vector landcover (GeoJSON) so backend venv need not include rasterio."""
    import geopandas as gpd
    from shapely.geometry import Polygon

    db = phase9_session
    lc_path = tmp_path / "lc.geojson"
    gpd.GeoDataFrame(
        {"landcover_class": ["CROPLAND"]},
        geometry=[
            Polygon([(76.99, 27.99), (77.03, 27.99), (77.03, 28.03), (76.99, 28.03)])
        ],
        crs="EPSG:4326",
    ).to_file(lc_path, driver="GeoJSON")
    cfg = _missing_cfg(
        landcover_raster_path=None,
        landcover_vector_path=lc_path,
        landcover_year="2020",
        landcover_source_name="test_landcover",
    )
    t0 = datetime(2026, 9, 5, 7, 0, tzinfo=timezone.utc)
    _insert_facility(db, fid_suffix="L", lat=28.01, lon=77.01)
    obs = _insert_obs(db, lat=28.01, lon=77.01, acq_datetime=t0, uniq="l1")
    process_one_observation(db, obs)
    eid = obs.event_id
    r1 = refresh_event_environmental(db, eid, config=cfg)
    ev = db.scalar(select(ThermalEvent).where(ThermalEvent.event_id == eid))
    assert ev.landcover_available is True
    assert ev.dominant_landcover_class == "CROPLAND"
    assert ev.landcover_year == "2020"
    snap = (
        ev.landcover_available,
        ev.dominant_landcover_class,
        ev.dominant_landcover_fraction,
        ev.water_context_available,
        ev.anomaly_status,
        ev.sta_association_status,
    )
    r2 = refresh_event_environmental(db, eid, config=cfg)
    db.refresh(ev)
    assert (
        ev.landcover_available,
        ev.dominant_landcover_class,
        ev.dominant_landcover_fraction,
        ev.water_context_available,
        ev.anomaly_status,
        ev.sta_association_status,
    ) == snap
    assert r1.to_dict() == r2.to_dict()


@REQUIRES_POSTGIS
def test_no_i3_corruption(phase9_session, tmp_path: Path) -> None:
    db = phase9_session
    water = write_water_geojson(tmp_path / "water.geojson")
    cfg = _missing_cfg(water_path=water)
    t0 = datetime(2026, 9, 5, 8, 0, tzinfo=timezone.utc)
    fac = _insert_facility(db, fid_suffix="F", lat=28.01, lon=77.01)
    # Seed a prior historical confirmed association so fingerprint exists
    hist = ThermalEvent(
        event_id=f"{EVT_PREFIX}PRIOR",
        event_start=datetime(2023, 6, 1, tzinfo=timezone.utc),
        event_end=datetime(2023, 6, 1, 2, tzinfo=timezone.utc),
        centroid_latitude=28.01,
        centroid_longitude=77.01,
        facility_id=fac.facility_id,
        facility_association_method="NEAR_FACILITY",
        is_active=False,
        detection_count=5,
        peak_frp=10.0,
        mean_frp=8.0,
        median_frp=8.0,
        observed_duration_hours=2.0,
        facility_distance_km=0.1,
    )
    db.add(hist)
    db.flush()
    obs = _insert_obs(db, lat=28.01, lon=77.01, acq_datetime=t0, uniq="f1")
    process_one_observation(db, obs)
    fp_before = db.scalar(
        select(func.count()).select_from(FacilityThermalFingerprint).where(
            FacilityThermalFingerprint.facility_id == fac.facility_id
        )
    )
    cur = db.scalar(select(ThermalEvent).where(ThermalEvent.event_id == obs.event_id))
    refresh_event_environmental(db, cur.event_id, config=cfg)
    fp_after = db.scalar(
        select(func.count()).select_from(FacilityThermalFingerprint).where(
            FacilityThermalFingerprint.facility_id == fac.facility_id
        )
    )
    assert fp_before == fp_after


@REQUIRES_POSTGIS
def test_no_fabricated_zeros_when_missing(phase9_session) -> None:
    db = phase9_session
    t0 = datetime(2026, 9, 5, 9, 0, tzinfo=timezone.utc)
    _insert_facility(db, fid_suffix="Z", lat=-41.1, lon=-171.1)
    obs = _insert_obs(db, lat=-41.1, lon=-171.1, acq_datetime=t0, uniq="z1")
    process_one_observation(db, obs)
    ev = db.scalar(select(ThermalEvent).where(ThermalEvent.event_id == obs.event_id))
    assert ev.landcover_available is False
    assert ev.dominant_landcover_fraction is None
    assert ev.vegetation_coverage_fraction is None
    assert ev.distance_to_builtup_km is None
    assert ev.satellite_value is None
    # I.6 must not invent environmental evidence; I.7/risk may still populate
    # fusion/risk columns from unavailable-domain semantics (e.g. score 0).
    assert ev.source_intelligence_candidate is not None
    assert ev.candidate_is_ground_truth is False
    assert ev.risk_score is not None
