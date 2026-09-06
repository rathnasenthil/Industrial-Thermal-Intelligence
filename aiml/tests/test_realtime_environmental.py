"""
Phase 9: realtime I.6 must match batch environmental-context semantics.

Does not invent environmental values when sources are missing.
Batch I.6 is spatial context only (no temporal matching of environmental layers).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from realtime.environmental import (
    process_event_environmental,
    unavailable_environmental_result,
)
from src.environmental_context.config import EnvironmentalContextConfig
from src.environmental_context.context_pipeline import run_environmental_context
from src.environmental_context.context_schema import ALL_CONTEXT_COLUMNS
from tests.fixtures.environmental_context.make_fixtures import (
    make_synthetic_events,
    write_landcover_raster,
    write_water_geojson,
)


def _missing_config(**overrides) -> EnvironmentalContextConfig:
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


def test_schema_parity_with_batch() -> None:
    assert "landcover_available" in ALL_CONTEXT_COLUMNS
    assert "satellite_value_name" in ALL_CONTEXT_COLUMNS
    assert len(ALL_CONTEXT_COLUMNS) == 26


def test_environmental_source_unavailable() -> None:
    events = make_synthetic_events()
    row = events.loc[events["event_id"] == "EVT_A"]
    result = process_event_environmental(row, "EVT_A", config=_missing_config())
    assert result.source_missing is True
    assert result.landcover_available is False
    assert result.vegetation_context_available is False
    assert result.builtup_context_available is False
    assert result.water_context_available is False
    assert result.agriculture_context_available is False
    assert result.satellite_context_available is False
    assert result.dominant_landcover_class is None
    assert result.dominant_landcover_fraction is None
    assert result.vegetation_present is None
    assert result.water_present is None
    assert result.satellite_value is None


def test_unavailable_helper_matches_empty_like() -> None:
    result = unavailable_environmental_result("X", source_missing=True)
    assert result.event_id == "X"
    assert result.landcover_available is False
    assert result.water_coverage_fraction is None


def test_landcover_available_with_fixture(tmp_path: Path) -> None:
    raster = write_landcover_raster(tmp_path / "lc.tif")
    events = make_synthetic_events()
    cfg = _missing_config(
        landcover_raster_path=raster,
        landcover_vector_path=None,
        landcover_class_map={"5": "CROPLAND", "2": "SHRUB"},
        landcover_year="2020",
    )
    result = process_event_environmental(
        events.loc[events["event_id"] == "EVT_A"], "EVT_A", config=cfg
    )
    assert result.source_missing is False
    assert result.landcover_available is True
    assert result.dominant_landcover_class in ("CROPLAND", "SHRUB", "5", "2")
    assert result.dominant_landcover_fraction == 1.0
    assert result.landcover_year == "2020"
    # Other domains still unavailable (paths missing)
    assert result.water_context_available is False
    assert result.water_present is None


def test_no_environmental_match_far_from_water(tmp_path: Path) -> None:
    water = write_water_geojson(tmp_path / "water.geojson")
    events = make_synthetic_events()
    cfg = _missing_config(water_path=water, context_buffer_km=1.0, broad_context_buffer_km=5.0)
    far = process_event_environmental(
        events.loc[events["event_id"] == "EVT_B"], "EVT_B", config=cfg
    )
    assert far.water_context_available is True
    assert far.water_present is False
    assert far.water_coverage_fraction == 0.0
    # Distance null when outside broad buffer (batch semantics)
    assert far.distance_to_water_km is None


def test_spatial_boundary_water_present(tmp_path: Path) -> None:
    water = write_water_geojson(tmp_path / "water.geojson")
    events = make_synthetic_events()
    cfg = _missing_config(water_path=water, context_buffer_km=1.0, broad_context_buffer_km=5.0)
    near = process_event_environmental(
        events.loc[events["event_id"] == "EVT_A"], "EVT_A", config=cfg
    )
    assert near.water_context_available is True
    assert near.water_present is True
    assert near.water_coverage_fraction is not None
    assert near.water_coverage_fraction > 0.0


def test_batch_realtime_parity(tmp_path: Path) -> None:
    raster = write_landcover_raster(tmp_path / "lc.tif")
    water = write_water_geojson(tmp_path / "water.geojson")
    events = make_synthetic_events()
    cfg = _missing_config(
        landcover_raster_path=raster,
        landcover_vector_path=None,
        water_path=water,
        landcover_class_map={"5": "CROPLAND", "2": "SHRUB"},
        landcover_year="2020",
    )
    batch = run_environmental_context(events, cfg)
    for eid in ("EVT_A", "EVT_B"):
        rt = process_event_environmental(
            events.loc[events["event_id"] == eid], eid, config=cfg
        )
        brow = batch.events_df.loc[batch.events_df["event_id"].astype(str) == eid].iloc[0]
        for col in ALL_CONTEXT_COLUMNS:
            left = rt.to_dict()[col]
            right = brow[col]
            if isinstance(right, float) and pd.isna(right):
                assert left is None
            elif isinstance(right, (bool, int)):
                assert bool(left) == bool(right)
            elif left is None:
                assert right is None or (isinstance(right, float) and pd.isna(right))
            elif isinstance(left, float) or (
                isinstance(right, (int, float)) and not isinstance(right, bool)
            ):
                assert left == pytest.approx(float(right), rel=1e-9, abs=1e-12)
            else:
                assert str(left) == str(right)


def test_idempotent_repeated_processing(tmp_path: Path) -> None:
    water = write_water_geojson(tmp_path / "water.geojson")
    events = make_synthetic_events()
    cfg = _missing_config(water_path=water)
    row = events.loc[events["event_id"] == "EVT_A"]
    r1 = process_event_environmental(row, "EVT_A", config=cfg)
    r2 = process_event_environmental(row, "EVT_A", config=cfg)
    assert r1.to_dict() == r2.to_dict()


def test_current_event_only_does_not_require_siblings(tmp_path: Path) -> None:
    water = write_water_geojson(tmp_path / "water.geojson")
    events = make_synthetic_events()
    cfg = _missing_config(water_path=water)
    # Only EVT_A row — must not need EVT_B in the frame
    only_a = events.loc[events["event_id"] == "EVT_A"]
    result = process_event_environmental(only_a, "EVT_A", config=cfg)
    assert result.event_id == "EVT_A"
    assert result.water_context_available is True


def test_i4_i5_fields_not_mutated_by_pipeline(tmp_path: Path) -> None:
    water = write_water_geojson(tmp_path / "water.geojson")
    events = make_synthetic_events()
    cfg = _missing_config(water_path=water)
    before = events.loc[events["event_id"] == "EVT_A"].iloc[0]
    result = process_event_environmental(
        events.loc[events["event_id"] == "EVT_A"], "EVT_A", config=cfg
    )
    # Adapter returns only I.6 fields; batch pipeline preserves I.4/I.5 on frame.
    batch = run_environmental_context(events.loc[events["event_id"] == "EVT_A"], cfg)
    after = batch.events_df.iloc[0]
    assert after["anomaly_status"] == before["anomaly_status"]
    assert after["sta_association_status"] == before["sta_association_status"]
    assert result.water_present is True


def test_no_fabricated_evidence_when_missing() -> None:
    events = make_synthetic_events()
    result = process_event_environmental(
        events.loc[events["event_id"] == "EVT_A"], "EVT_A", config=_missing_config()
    )
    # Missing evidence must not become zeros for landcover/satellite/distances
    assert result.dominant_landcover_fraction is None
    assert result.landcover_class_count is None
    assert result.distance_to_vegetation_km is None
    assert result.satellite_value is None
