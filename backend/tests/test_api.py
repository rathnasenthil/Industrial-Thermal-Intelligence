"""API route tests using dependency/service overrides (no PostGIS required)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas.dashboard import DashboardStatistics
from app.schemas.events import (
    EventDetail,
    EventEvidence,
    EventSummary,
    EventTimeline,
    EvidenceFamilyBlock,
    PaginatedAlerts,
    PaginatedEvents,
)
from app.schemas.facilities import FacilityDetail, FacilitySummary, PaginatedFacilities

client = TestClient(app)


def _sample_summary() -> EventSummary:
    return EventSummary(
        event_id="EVT_0000001",
        event_start=datetime(2023, 1, 1, 6, 55, tzinfo=timezone.utc),
        event_end=datetime(2023, 1, 1, 6, 55, tzinfo=timezone.utc),
        observed_duration_hours=0.0,
        detection_count=2,
        peak_frp=2.41,
        mean_frp=2.345,
        latitude=18.97,
        longitude=83.80,
        persistence_label="EPHEMERAL",
        facility_type="OTHER_INDUSTRIAL",
        anomaly_status="INSUFFICIENT_HISTORY",
        industrial_context="INSUFFICIENT_EVIDENCE",
        risk_score=10.0,
        investigation_priority="LOW",
        thermal_severity_band="LOW",
        recommended_action="MONITOR",
    )


def test_health_returns_ok() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_events_pagination_and_filters() -> None:
    payload = PaginatedEvents(
        items=[_sample_summary()],
        total=1,
        page=1,
        page_size=50,
        total_pages=1,
    )
    with patch("app.api.routes.events.events_service.list_events", return_value=payload):
        response = client.get(
            "/api/events",
            params={
                "page": 1,
                "page_size": 50,
                "priority": "LOW",
                "industrial_context": "INSUFFICIENT_EVIDENCE",
                "persistence_class": "EPHEMERAL",
                "anomaly_status": "INSUFFICIENT_HISTORY",
                "min_risk_score": 0,
                "max_risk_score": 50,
                "date_from": "2023-01-01T00:00:00Z",
                "date_to": "2023-12-31T23:59:59Z",
                "bbox": "72,8,97,37",
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["event_id"] == "EVT_0000001"
    assert "probability" not in body["items"][0]


def test_event_detail_404() -> None:
    with patch("app.api.routes.events.events_service.get_event", return_value=None):
        response = client.get("/api/events/MISSING")
    assert response.status_code == 404


def test_event_detail_ok() -> None:
    detail = EventDetail(**_sample_summary().model_dump())
    with patch("app.api.routes.events.events_service.get_event", return_value=detail):
        response = client.get("/api/events/EVT_0000001")
    assert response.status_code == 200
    assert "decision-support" in response.json()["semantics_note"]


def test_event_evidence_marks_sta_unavailable() -> None:
    evidence = EventEvidence(
        event_id="EVT_0000001",
        temporal=EvidenceFamilyBlock(available=True, status="available", score=1.0),
        infrastructure=EvidenceFamilyBlock(available=True, status="available", score=1.0),
        historical=EvidenceFamilyBlock(available=False, status="unavailable"),
        anomaly=EvidenceFamilyBlock(available=False, status="unavailable"),
        sta=EvidenceFamilyBlock(
            available=False,
            status="unavailable",
            summary="STA domain unavailable",
        ),
        environmental=EvidenceFamilyBlock(
            available=False,
            status="unavailable",
            summary="Environmental domain unavailable",
        ),
        fusion={"candidate_is_ground_truth": False},
    )
    with patch(
        "app.api.routes.events.events_service.get_event_evidence",
        return_value=evidence,
    ):
        response = client.get("/api/events/EVT_0000001/evidence")
    assert response.status_code == 200
    body = response.json()
    assert body["sta"]["status"] == "unavailable"
    assert body["environmental"]["status"] == "unavailable"
    assert body["sta"]["score"] is None


def test_event_timeline_does_not_fabricate_detections() -> None:
    timeline = EventTimeline(
        event_id="EVT_0000001",
        detection_count=2,
        detection_level_timeline_available=False,
    )
    with patch(
        "app.api.routes.events.events_service.get_event_timeline",
        return_value=timeline,
    ):
        response = client.get("/api/events/EVT_0000001/timeline")
    assert response.status_code == 200
    assert response.json()["detection_level_timeline_available"] is False


def test_facilities_list_and_detail() -> None:
    facilities = PaginatedFacilities(
        items=[
            FacilitySummary(
                facility_id="osm_node_1",
                facility_name="Plant",
                facility_type="POWER_PLANT",
                latitude=11.3,
                longitude=76.7,
            )
        ],
        total=1,
        page=1,
        page_size=50,
        total_pages=1,
    )
    detail = FacilityDetail(
        facility_id="osm_node_1",
        facility_name="Plant",
        facility_type="POWER_PLANT",
        latitude=11.3,
        longitude=76.7,
        source="osm_static_extract",
    )
    with patch(
        "app.api.routes.facilities.facilities_service.list_facilities",
        return_value=facilities,
    ):
        response = client.get("/api/facilities", params={"search": "Plant", "bbox": "70,8,80,15"})
    assert response.status_code == 200
    assert response.json()["items"][0]["facility_id"] == "osm_node_1"

    with patch(
        "app.api.routes.facilities.facilities_service.get_facility",
        return_value=detail,
    ):
        response = client.get("/api/facilities/osm_node_1")
    assert response.status_code == 200

    with patch(
        "app.api.routes.facilities.facilities_service.get_facility",
        return_value=None,
    ):
        response = client.get("/api/facilities/missing")
    assert response.status_code == 404


def test_alerts_are_high_critical_view() -> None:
    payload = PaginatedAlerts(
        items=[_sample_summary()],
        total=1,
        page=1,
        page_size=50,
        total_pages=1,
    )
    with patch("app.api.routes.alerts.events_service.list_alerts", return_value=payload) as mock:
        response = client.get("/api/alerts")
    assert response.status_code == 200
    mock.assert_called_once()


def test_dashboard_statistics() -> None:
    stats = DashboardStatistics(
        total_events=10,
        total_facilities=5,
        priority_distribution={"LOW": 8, "HIGH": 2},
        industrial_context_distribution={"INSUFFICIENT_EVIDENCE": 10},
        persistence_distribution={"EPHEMERAL": 10},
        thermal_severity_distribution={"LOW": 10},
        anomaly_distribution={"INSUFFICIENT_HISTORY": 10},
        facility_type_distribution={"POWER_PLANT": 3},
        events_with_facility_association=3,
        events_without_facility_association=7,
        high_priority_count=2,
        critical_count=0,
    )
    with patch(
        "app.api.routes.dashboard.dashboard_service.get_dashboard_statistics",
        return_value=stats,
    ):
        response = client.get("/api/dashboard/statistics")
    assert response.status_code == 200
    body = response.json()
    assert body["total_events"] == 10
    assert "validated performance" in body["semantics_note"].lower() or "validated" in body[
        "semantics_note"
    ].lower()


def test_events_invalid_bbox_returns_400() -> None:
    with patch(
        "app.api.routes.events.events_service.list_events",
        side_effect=ValueError("bbox must be min_lon,min_lat,max_lon,max_lat"),
    ):
        response = client.get("/api/events", params={"bbox": "bad"})
    assert response.status_code == 400
