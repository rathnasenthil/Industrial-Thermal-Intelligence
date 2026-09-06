"""Integration tests requiring a live PostGIS database and ingested data."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import func, select, text

from app.models.facility import Facility
from app.models.thermal_event import ThermalEvent
from app.services.dashboard import get_dashboard_statistics
from app.services.events import get_event, list_events
from app.services.facilities import get_facility, list_facilities
from app.services.ingestion import run_ingestion
from tests.conftest import REQUIRES_POSTGIS

REPO_ROOT = Path(__file__).resolve().parents[2]
EVENTS_CSV = REPO_ROOT / "aiml/data/processed/thermal_events_with_risk_prioritization.csv"
FACILITIES_CSV = REPO_ROOT / "aiml/data/processed/osm_facilities.csv"
CANDIDATES_CSV = REPO_ROOT / "aiml/data/processed/thermal_event_facility_candidates.csv"

EXPECTED_PRODUCTION_EVENTS = 179_740
EXPECTED_PRODUCTION_FACILITIES = 112_956


@REQUIRES_POSTGIS
@pytest.mark.slow
def test_ingestion_row_counts_match_source(db_session) -> None:
    if not EVENTS_CSV.exists() or not FACILITIES_CSV.exists():
        pytest.skip("Stage VI / I.1 CSVs not present on disk")

    report = run_ingestion(
        db_session,
        events_csv=EVENTS_CSV,
        facilities_csv=FACILITIES_CSV,
        candidates_csv=CANDIDATES_CSV,
        mode="replace",
        load_candidates=True,
    )
    assert not report.errors, report.errors
    assert report.events_inserted == report.events_source_rows - report.events_rejected
    assert report.events_rejected == 0
    assert report.facilities_rejected == 0

    db_events = db_session.scalar(select(func.count()).select_from(ThermalEvent))
    db_facilities = db_session.scalar(select(func.count()).select_from(Facility))
    assert db_events == report.events_inserted
    assert db_facilities == report.facilities_inserted

    # Production expectation used only as an ingestion validation check.
    assert report.events_source_rows == EXPECTED_PRODUCTION_EVENTS
    assert report.facilities_source_rows == EXPECTED_PRODUCTION_FACILITIES

    srid = db_session.execute(
        text(
            "SELECT ST_SRID(geometry) FROM thermal_events "
            "WHERE geometry IS NOT NULL LIMIT 1"
        )
    ).scalar()
    assert int(srid) == 4326

    null_sta = db_session.scalar(
        select(func.count()).where(ThermalEvent.sta_domain_available.is_(False))
    )
    assert null_sta is not None
    assert null_sta > 0


@REQUIRES_POSTGIS
def test_api_queries_against_db_if_populated(db_session) -> None:
    count = db_session.scalar(select(func.count()).select_from(ThermalEvent)) or 0
    if count == 0:
        pytest.skip("thermal_events empty — run ingestion first")

    page = list_events(db_session, page=1, page_size=10, priority="CRITICAL")
    assert page.page_size == 10
    assert all(item.investigation_priority == "CRITICAL" for item in page.items)

    boxed = list_events(db_session, page=1, page_size=5, bbox="68,6,98,38")
    assert boxed.total >= 0

    sample_id = db_session.scalar(select(ThermalEvent.event_id).limit(1))
    detail = get_event(db_session, sample_id)
    assert detail is not None
    assert detail.event_id == sample_id

    facilities = list_facilities(db_session, page=1, page_size=5)
    assert facilities.total > 0
    facility = get_facility(db_session, facilities.items[0].facility_id)
    assert facility is not None

    stats = get_dashboard_statistics(db_session)
    assert stats.total_events == count
    assert sum(stats.priority_distribution.values()) <= stats.total_events
    assert stats.critical_count == stats.priority_distribution.get("CRITICAL", 0)
    assert stats.high_priority_count == stats.priority_distribution.get("HIGH", 0)
