"""Thin module tests for remaining I.6 context sources."""

from __future__ import annotations

from pathlib import Path

from src.environmental_context.agriculture_context import compute_agriculture_context
from src.environmental_context.builtup_context import compute_builtup_context
from src.environmental_context.config import EnvironmentalContextConfig
from src.environmental_context.satellite_context import compute_satellite_context
from tests.fixtures.environmental_context.make_fixtures import make_synthetic_events


def test_builtup_missing() -> None:
    frame, meta = compute_builtup_context(
        make_synthetic_events(), EnvironmentalContextConfig(builtup_path=Path("missing.geojson"))
    )
    assert meta["available"] is False
    assert frame["builtup_coverage_fraction"].isna().all()


def test_agriculture_missing() -> None:
    frame, meta = compute_agriculture_context(
        make_synthetic_events(), EnvironmentalContextConfig(agriculture_path=Path("missing.geojson"))
    )
    assert meta["available"] is False
    assert frame["agriculture_present"].isna().all()


def test_satellite_missing() -> None:
    frame, meta = compute_satellite_context(
        make_synthetic_events(), EnvironmentalContextConfig(satellite_raster_path=Path("missing.tif"))
    )
    assert meta["available"] is False
    assert frame["satellite_context_available"].eq(False).all()
    assert frame["satellite_value"].isna().all()
