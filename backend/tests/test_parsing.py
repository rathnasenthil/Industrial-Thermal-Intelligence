"""Unit tests for CSV parsing helpers (no database required)."""

from datetime import timezone

from app.schemas.common import parse_bbox
from app.services.ingestion import _candidate_row, _event_row, _facility_row
from app.services.parsing import (
    parse_optional_bool,
    parse_optional_float,
    parse_timestamp,
    valid_lon_lat,
)


def test_parse_timestamp_preserves_utc() -> None:
    dt = parse_timestamp("2023-01-01T06:55:00+00:00")
    assert dt is not None
    assert dt.tzinfo is not None
    assert dt.astimezone(timezone.utc).hour == 6


def test_parse_optional_float_preserves_nan_as_none() -> None:
    assert parse_optional_float("") is None
    assert parse_optional_float("nan") is None
    assert parse_optional_float("2.41") == 2.41


def test_parse_optional_bool() -> None:
    assert parse_optional_bool("False") is False
    assert parse_optional_bool("true") is True
    assert parse_optional_bool("") is None


def test_valid_lon_lat() -> None:
    assert valid_lon_lat(83.8, 18.9) is True
    assert valid_lon_lat(200.0, 18.9) is False
    assert valid_lon_lat(None, 18.9) is False


def test_parse_bbox() -> None:
    box = parse_bbox("72.0,8.0,97.0,37.0")
    assert box.min_lon == 72.0
    assert box.max_lat == 37.0


def test_event_row_preserves_null_anomaly_score() -> None:
    raw = {
        "event_id": "EVT_TEST_1",
        "event_start": "2023-01-01T06:55:00+00:00",
        "event_end": "2023-01-01T06:55:00+00:00",
        "centroid_latitude": "18.97",
        "centroid_longitude": "83.80",
        "detection_count": "2",
        "peak_frp": "2.41",
        "mean_frp": "2.3",
        "persistence_label": "EPHEMERAL",
        "risk_score": "12.5",
        "investigation_priority": "LOW",
        "industrial_context": "INSUFFICIENT_EVIDENCE",
        "recommended_action": "MONITOR",
        "anomaly_score": "",
        "sta_domain_available": "False",
        "environmental_domain_available": "False",
        "sta_evidence_score": "",
        "facility_id": "",
    }
    row, err = _event_row(raw)
    assert err is None
    assert row is not None
    assert row["anomaly_score"] is None
    assert row["sta_evidence_score"] is None
    assert row["facility_id"] is None
    assert row["sta_domain_available"] is False
    assert row["geometry"] is not None


def test_event_row_rejects_out_of_range_risk() -> None:
    raw = {
        "event_id": "EVT_BAD",
        "centroid_latitude": "18.97",
        "centroid_longitude": "83.80",
        "risk_score": "150",
    }
    row, err = _event_row(raw)
    assert row is None
    assert err is not None
    assert "risk_score" in err


def test_facility_row_and_candidate_row() -> None:
    facility, err = _facility_row(
        {
            "facility_id": "osm_node_1",
            "facility_name": "Test Plant",
            "facility_type": "POWER_PLANT",
            "latitude": "11.3",
            "longitude": "76.7",
            "osm_id": "1",
            "osm_type": "node",
            "source": "osm",
            "source_version": "v1",
            "osm_tags": '{"name": "Test Plant"}',
        }
    )
    assert err is None
    assert facility is not None
    assert facility["osm_tags"]["name"] == "Test Plant"

    cand, cerr = _candidate_row(
        {
            "event_id": "EVT_1",
            "facility_id": "osm_node_1",
            "spatial_relation": "NEAR_FACILITY",
            "distance_km": "4.5",
            "candidate_rank": "1",
            "candidate_score": "-4.5",
        }
    )
    assert cerr is None
    assert cand is not None
    assert cand["candidate_rank"] == 1
