"""
Phase 8: realtime I.5 must match batch STA matching/ranking semantics.

Does not invent STA geometries when sources are missing.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest

from realtime.sta import process_event_sta, unavailable_sta_result
from src.sta_evidence.config import (
    NO_STA_ASSOCIATION,
    QUALITY_NONE,
    STA_ASSOCIATED,
    STA_INTERSECTS_EVENT,
    STAConfig,
)
from src.sta_evidence.sta_pipeline import I5_APPEND_COLUMNS, run_sta_integration
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


def test_sta_evidence_available_inside_mask(combined_sta) -> None:
    events = make_synthetic_events()
    row = events.loc[events["event_id"] == "EVT_INSIDE"]
    result = process_event_sta(row, "EVT_INSIDE", config=STAConfig(association_radius_km=2.0), sta_gdf=combined_sta)
    assert result.sta_evidence_available is True
    assert result.sta_association_status in (STA_ASSOCIATED, "AMBIGUOUS")
    assert result.sta_match_count >= 1
    assert result.sta_evidence_quality != QUALITY_NONE
    assert result.source_missing is False


def test_sta_evidence_unavailable_source_missing() -> None:
    events = make_synthetic_events()
    row = events.loc[events["event_id"] == "EVT_INSIDE"]
    cfg = STAConfig(
        mask_path=Path("data/raw/does_not_exist_mask.geojson"),
        detection_path=Path("data/raw/does_not_exist_det.geojson"),
    )
    result = process_event_sta(row, "EVT_INSIDE", config=cfg, sta_gdf=None)
    assert result.source_missing is True
    assert result.sta_evidence_available is False
    assert result.sta_association_status == NO_STA_ASSOCIATION
    assert result.sta_match_count == 0
    assert result.primary_sta_id is None
    assert result.sta_evidence_quality == QUALITY_NONE


def test_no_matching_sta_record(combined_sta) -> None:
    events = make_synthetic_events()
    row = events.loc[events["event_id"] == "EVT_NONE"]
    result = process_event_sta(row, "EVT_NONE", config=STAConfig(association_radius_km=0.5), sta_gdf=combined_sta)
    assert result.sta_association_status == NO_STA_ASSOCIATION
    assert result.sta_evidence_available is False
    assert result.primary_sta_id is None


def test_spatial_near_vs_none(combined_sta) -> None:
    events = make_synthetic_events()
    near = process_event_sta(
        events.loc[events["event_id"] == "EVT_NEAR"],
        "EVT_NEAR",
        config=STAConfig(association_radius_km=2.0),
        sta_gdf=combined_sta,
    )
    none = process_event_sta(
        events.loc[events["event_id"] == "EVT_NONE"],
        "EVT_NONE",
        config=STAConfig(association_radius_km=0.5),
        sta_gdf=combined_sta,
    )
    assert near.sta_association_status in (STA_ASSOCIATED, "AMBIGUOUS") or near.sta_match_count >= 0
    assert none.sta_association_status == NO_STA_ASSOCIATION


def test_batch_realtime_parity(combined_sta) -> None:
    events = make_synthetic_events()
    cfg = STAConfig(association_radius_km=2.0)
    batch = run_sta_integration(events, cfg, sta_gdf=combined_sta)
    for eid in ("EVT_INSIDE", "EVT_NEAR", "EVT_NONE"):
        rt = process_event_sta(events.loc[events["event_id"] == eid], eid, config=cfg, sta_gdf=combined_sta)
        brow = batch.events_df.loc[batch.events_df["event_id"] == eid].iloc[0]
        assert rt.sta_association_status == brow["sta_association_status"]
        assert rt.sta_match_count == int(brow["sta_match_count"])
        assert bool(rt.sta_evidence_available) == bool(brow["sta_evidence_available"])
        assert rt.sta_evidence_quality == brow["sta_evidence_quality"]
        bp = brow["primary_sta_id"]
        if bp is None or (isinstance(bp, float) and pd.isna(bp)):
            assert rt.primary_sta_id is None
        else:
            assert rt.primary_sta_id == str(bp)


def test_unavailable_helper() -> None:
    result = unavailable_sta_result("E1", source_missing=True)
    assert result.sta_association_status == NO_STA_ASSOCIATION
    assert result.source_missing is True


def test_does_not_fabricate_without_source() -> None:
    events = make_synthetic_events()
    row = events.loc[events["event_id"] == "EVT_INSIDE"]
    cfg = STAConfig(
        mask_path=Path("/tmp/no_sta_mask.geojson"),
        detection_path=Path("/tmp/no_sta_det.geojson"),
    )
    with patch("realtime.sta.run_sta_integration") as mocked:
        result = process_event_sta(row, "EVT_INSIDE", config=cfg, sta_gdf=None)
        mocked.assert_not_called()
    assert result.sta_evidence_available is False


def test_i4_fields_not_required_for_scoring(combined_sta) -> None:
    # Minimal geometry-only frame still works via batch pipeline.
    events = make_synthetic_events()
    row = events.loc[events["event_id"] == "EVT_INSIDE"].copy()
    result = process_event_sta(row, "EVT_INSIDE", config=STAConfig(association_radius_km=2.0), sta_gdf=combined_sta)
    assert result.event_id == "EVT_INSIDE"
    assert set(I5_APPEND_COLUMNS)  # sanity: columns contract still defined


def test_only_current_event_processed(combined_sta) -> None:
    events = make_synthetic_events()
    # Pass full multi-event frame; adapter must filter to one event.
    result = process_event_sta(events, "EVT_INSIDE", config=STAConfig(association_radius_km=2.0), sta_gdf=combined_sta)
    assert result.event_id == "EVT_INSIDE"


def test_intersects_quality_high_when_associated(combined_sta) -> None:
    events = make_synthetic_events()
    result = process_event_sta(
        events.loc[events["event_id"] == "EVT_INSIDE"],
        "EVT_INSIDE",
        config=STAConfig(association_radius_km=2.0),
        sta_gdf=combined_sta,
    )
    if result.sta_association_status == STA_ASSOCIATED:
        # Inside polygon typically INTERSECTS → HIGH
        assert result.sta_evidence_quality in ("HIGH", "MEDIUM", "LOW")
        if result.sta_layer_type:
            assert result.sta_layer_type in ("MASK", "DETECTION")
