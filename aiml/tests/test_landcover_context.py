"""Tests for land-cover / water / vegetation context modules."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.environmental_context.config import EnvironmentalContextConfig
from src.environmental_context.landcover_context import compute_landcover_context
from src.environmental_context.vegetation_context import compute_vegetation_context
from src.environmental_context.water_context import compute_water_context
from tests.fixtures.environmental_context.make_fixtures import (
    make_synthetic_events,
    write_landcover_raster,
    write_water_geojson,
)


def test_missing_landcover_unavailable() -> None:
    events = make_synthetic_events()
    config = EnvironmentalContextConfig(
        landcover_raster_path=Path("data/external/does_not_exist.tif"),
        landcover_vector_path=Path("data/external/does_not_exist.geojson"),
    )
    frame, meta = compute_landcover_context(events, config)
    assert meta["available"] is False
    assert frame["landcover_available"].eq(False).all()
    assert frame["dominant_landcover_fraction"].isna().all()


def test_landcover_raster_categorical(tmp_path: Path) -> None:
    raster = write_landcover_raster(tmp_path / "lc.tif")
    events = make_synthetic_events()
    config = EnvironmentalContextConfig(
        landcover_raster_path=raster,
        landcover_vector_path=None,
        landcover_class_map={"5": "CROPLAND", "2": "SHRUB"},
        landcover_year="2020",
    )
    frame, meta = compute_landcover_context(events, config)
    assert meta["available"] is True
    row = frame.loc[frame["event_id"] == "EVT_A"].iloc[0]
    assert row["landcover_available"] in (True, 1)
    assert row["dominant_landcover_class"] in ("CROPLAND", "SHRUB", "5", "2")
    assert row["dominant_landcover_fraction"] == 1.0


def test_water_vector_presence_and_distance(tmp_path: Path) -> None:
    water = write_water_geojson(tmp_path / "water.geojson")
    events = make_synthetic_events()
    config = EnvironmentalContextConfig(water_path=water, context_buffer_km=1.0, broad_context_buffer_km=5.0)
    frame, meta = compute_water_context(events, config)
    assert meta["available"] is True
    a = frame.loc[frame["event_id"] == "EVT_A"].iloc[0]
    assert a["water_context_available"] in (True, 1)
    assert a["water_present"] in (True, 1)
    b = frame.loc[frame["event_id"] == "EVT_B"].iloc[0]
    # EVT_B is far from the Delhi-area lake
    assert b["water_present"] in (False, 0)


def test_vegetation_missing_null_semantics() -> None:
    events = make_synthetic_events()
    config = EnvironmentalContextConfig(vegetation_path=Path("nope.geojson"))
    frame, meta = compute_vegetation_context(events, config)
    assert meta["available"] is False
    assert frame["vegetation_coverage_fraction"].isna().all()
    assert frame["distance_to_vegetation_km"].isna().all()
    assert frame["vegetation_present"].isna().all()
