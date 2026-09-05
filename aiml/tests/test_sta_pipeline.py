"""Integration tests for Stage I.5 STA pipeline."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from src.sta_evidence.config import NO_STA_ASSOCIATION, STA_ASSOCIATED, STAConfig
from src.sta_evidence.sta_pipeline import I4_IMMUTABLE_COLUMNS, run_sta_integration, save_outputs
from tests.fixtures.sta.make_fixtures import (
    load_det_as_gdf,
    load_mask_as_gdf,
    make_synthetic_events,
    write_synthetic_sta_detections_geojson,
    write_synthetic_sta_mask_geojson,
)


@pytest.fixture()
def combined_sta(tmp_path: Path):
    mask = write_synthetic_sta_mask_geojson(tmp_path / "mask.geojson")
    det = write_synthetic_sta_detections_geojson(tmp_path / "det.geojson")
    combined = pd.concat([load_mask_as_gdf(mask), load_det_as_gdf(det)], ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")


def test_event_count_preserved(combined_sta) -> None:
    events = make_synthetic_events()
    result = run_sta_integration(events, STAConfig(association_radius_km=2.0), sta_gdf=combined_sta)
    assert len(result.events_df) == len(events)
    assert result.events_df["event_id"].is_unique


def test_i4_fields_unchanged(combined_sta) -> None:
    events = make_synthetic_events()
    result = run_sta_integration(events, STAConfig(association_radius_km=2.0), sta_gdf=combined_sta)
    left = events.sort_values("event_id").reset_index(drop=True)
    right = result.events_df.sort_values("event_id").reset_index(drop=True)
    for col in I4_IMMUTABLE_COLUMNS:
        a = pd.to_numeric(left[col], errors="coerce") if col.endswith("deviation") or col == "anomaly_score" else left[col]
        b = pd.to_numeric(right[col], errors="coerce") if col.endswith("deviation") or col == "anomaly_score" else right[col]
        if col in ("anomaly_status", "anomaly_confidence"):
            assert list(left[col]) == list(right[col])
        else:
            assert a.fillna(-1e18).tolist() == b.fillna(-1e18).tolist()


def test_no_facility_still_gets_sta(combined_sta) -> None:
    events = make_synthetic_events()
    result = run_sta_integration(events, STAConfig(association_radius_km=2.0), sta_gdf=combined_sta)
    row = result.events_df.loc[result.events_df["event_id"] == "EVT_INSIDE"].iloc[0]
    assert row["facility_association_method"] == "NO_FACILITY_ASSOCIATION"
    assert row["sta_association_status"] in (STA_ASSOCIATED, "AMBIGUOUS")
    assert row["sta_evidence_available"] in (True, 1)


def test_ambiguous_facility_unchanged(combined_sta) -> None:
    events = make_synthetic_events()
    result = run_sta_integration(events, STAConfig(association_radius_km=2.0), sta_gdf=combined_sta)
    row = result.events_df.loc[result.events_df["event_id"] == "EVT_AMBIG_FAC"].iloc[0]
    assert row["facility_association_method"] == "AMBIGUOUS"
    assert pd.isna(row["facility_id"]) or row["facility_id"] is None or str(row["facility_id"]) == "nan"


def test_no_sta_association_preserved(combined_sta) -> None:
    events = make_synthetic_events()
    result = run_sta_integration(events, STAConfig(association_radius_km=0.5), sta_gdf=combined_sta)
    row = result.events_df.loc[result.events_df["event_id"] == "EVT_NONE"].iloc[0]
    assert row["sta_association_status"] == NO_STA_ASSOCIATION
    assert row["sta_evidence_quality"] == "NONE"
    assert pd.isna(row["primary_sta_id"]) or row["primary_sta_id"] is None


def test_candidates_preserved(combined_sta) -> None:
    events = make_synthetic_events()
    result = run_sta_integration(events, STAConfig(association_radius_km=2.0), sta_gdf=combined_sta)
    assert "event_id" in result.candidates_df.columns
    assert "sta_id" in result.candidates_df.columns
    assert "candidate_rank" in result.candidates_df.columns


def test_pipeline_deterministic(combined_sta) -> None:
    events = make_synthetic_events()
    r1 = run_sta_integration(events, STAConfig(association_radius_km=2.0), sta_gdf=combined_sta)
    r2 = run_sta_integration(events, STAConfig(association_radius_km=2.0), sta_gdf=combined_sta)
    cols = ["event_id", "sta_association_status", "primary_sta_id", "sta_match_count", "sta_evidence_quality"]
    pd.testing.assert_frame_equal(r1.events_df[cols].reset_index(drop=True), r2.events_df[cols].reset_index(drop=True))
    pd.testing.assert_frame_equal(r1.candidates_df.reset_index(drop=True), r2.candidates_df.reset_index(drop=True))


def test_no_forbidden_classification_fields(combined_sta) -> None:
    events = make_synthetic_events()
    result = run_sta_integration(events, STAConfig(association_radius_km=2.0), sta_gdf=combined_sta)
    forbidden = ("industrial_fire", "source_class", "wildfire_probability", "risk_score", "fire_type")
    blob = " ".join(result.events_df.columns).lower()
    for term in forbidden:
        assert term not in blob


def test_no_literal_nan_string(tmp_path: Path, combined_sta) -> None:
    events = make_synthetic_events()
    result = run_sta_integration(events, STAConfig(association_radius_km=2.0), sta_gdf=combined_sta)
    out = tmp_path / "events.csv"
    cand = tmp_path / "cand.csv"
    sta = tmp_path / "sta.csv"
    save_outputs(result, events_output_path=out, candidates_output_path=cand, sta_normalized_output_path=sta)
    text = out.read_text(encoding="utf-8")
    # Allow empty fields; reject standalone nan tokens as values (rough check on object cols)
    reloaded = pd.read_csv(out)
    for col in reloaded.select_dtypes(include=["object", "str"]).columns:
        assert not ((reloaded[col] == "nan") & reloaded[col].notna()).any()


def test_report_sections(combined_sta) -> None:
    events = make_synthetic_events()
    result = run_sta_integration(events, STAConfig(association_radius_km=2.0), sta_gdf=combined_sta)
    for key in (
        "input",
        "validation",
        "spatial_matching",
        "evidence_quality_counts",
        "candidate_statistics",
        "configuration",
        "i4_immutability",
        "limitations",
    ):
        assert key in result.report
    assert result.report["i4_immutability"]["anomaly_fields_unchanged"] is True
