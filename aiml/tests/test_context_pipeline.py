"""Integration tests for Stage I.6 environmental context pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.environmental_context.config import EnvironmentalContextConfig
from src.environmental_context.context_pipeline import run_environmental_context, save_outputs
from src.environmental_context.context_schema import I4_IMMUTABLE_COLUMNS
from tests.fixtures.environmental_context.make_fixtures import (
    make_synthetic_events,
    write_landcover_raster,
    write_water_geojson,
)


def test_one_row_per_event_and_ids_preserved() -> None:
    events = make_synthetic_events()
    config = EnvironmentalContextConfig(
        landcover_raster_path=Path("missing.tif"),
        landcover_vector_path=None,
        vegetation_path=None,
        builtup_path=None,
        water_path=None,
        agriculture_path=None,
        satellite_raster_path=None,
    )
    result = run_environmental_context(events, config)
    assert len(result.events_df) == len(events)
    assert result.events_df["event_id"].is_unique
    assert set(result.events_df["event_id"]) == set(events["event_id"])


def test_deterministic_ordering_and_repeat() -> None:
    events = make_synthetic_events()
    config = EnvironmentalContextConfig(
        landcover_raster_path=None,
        landcover_vector_path=None,
        vegetation_path=None,
        builtup_path=None,
        water_path=None,
        agriculture_path=None,
        satellite_raster_path=None,
    )
    r1 = run_environmental_context(events, config)
    r2 = run_environmental_context(events.sample(frac=1.0, random_state=0), config)
    assert list(r1.events_df["event_id"]) == list(r2.events_df["event_id"])
    pd.testing.assert_frame_equal(
        r1.events_df[["event_id", "landcover_available", "water_context_available"]],
        r2.events_df[["event_id", "landcover_available", "water_context_available"]],
    )


def test_i4_and_i5_fields_immutable(tmp_path: Path) -> None:
    events = make_synthetic_events()
    water = write_water_geojson(tmp_path / "water.geojson")
    config = EnvironmentalContextConfig(
        water_path=water,
        landcover_raster_path=None,
        landcover_vector_path=None,
        vegetation_path=None,
        builtup_path=None,
        agriculture_path=None,
        satellite_raster_path=None,
    )
    result = run_environmental_context(events, config)
    left = events.sort_values("event_id").reset_index(drop=True)
    right = result.events_df.sort_values("event_id").reset_index(drop=True)
    for col in I4_IMMUTABLE_COLUMNS:
        if col in ("anomaly_status", "anomaly_confidence"):
            assert list(left[col]) == list(right[col])
        else:
            assert pd.to_numeric(left[col], errors="coerce").fillna(-1).tolist() == pd.to_numeric(
                right[col], errors="coerce"
            ).fillna(-1).tolist()
    assert list(left["sta_association_status"]) == list(right["sta_association_status"])
    assert list(left["sta_evidence_quality"]) == list(right["sta_evidence_quality"])


def test_no_literal_nan(tmp_path: Path) -> None:
    events = make_synthetic_events()
    config = EnvironmentalContextConfig(
        landcover_raster_path=None,
        landcover_vector_path=None,
        vegetation_path=None,
        builtup_path=None,
        water_path=None,
        agriculture_path=None,
        satellite_raster_path=None,
    )
    result = run_environmental_context(events, config)
    out = tmp_path / "out.csv"
    save_outputs(result, out)
    reloaded = pd.read_csv(out)
    for col in reloaded.select_dtypes(include=["object", "str"]).columns:
        assert not ((reloaded[col] == "nan") & reloaded[col].notna()).any()


def test_no_source_classification_fields() -> None:
    events = make_synthetic_events()
    config = EnvironmentalContextConfig(
        landcover_raster_path=None,
        vegetation_path=None,
        builtup_path=None,
        water_path=None,
        agriculture_path=None,
        satellite_raster_path=None,
        landcover_vector_path=None,
    )
    result = run_environmental_context(events, config)
    blob = " ".join(result.events_df.columns).lower()
    for term in ("industrial_fire", "wildfire", "agricultural_fire", "risk_score", "source_class"):
        assert term not in blob


def test_report_distinguishes_available_and_missing(tmp_path: Path) -> None:
    events = make_synthetic_events()
    raster = write_landcover_raster(tmp_path / "lc.tif")
    config = EnvironmentalContextConfig(
        landcover_raster_path=raster,
        landcover_vector_path=None,
        vegetation_path=None,
        builtup_path=None,
        water_path=None,
        agriculture_path=None,
        satellite_raster_path=None,
    )
    result = run_environmental_context(events, config)
    assert result.report["per_source_availability"]["landcover"] is True
    assert result.report["per_source_availability"]["water"] is False
    assert "landcover" in result.report["datasets_detected"]
    assert "water" in result.report["datasets_missing"]


def test_events_dataframe_not_mutated() -> None:
    events = make_synthetic_events()
    original = events.copy(deep=True)
    config = EnvironmentalContextConfig(
        landcover_raster_path=None,
        landcover_vector_path=None,
        vegetation_path=None,
        builtup_path=None,
        water_path=None,
        agriculture_path=None,
        satellite_raster_path=None,
    )
    run_environmental_context(events, config)
    pd.testing.assert_frame_equal(events, original)
