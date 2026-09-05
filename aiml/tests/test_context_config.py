"""Tests for Stage I.6 configuration and schema."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.environmental_context.config import EnvironmentalContextConfig
from src.environmental_context.context_schema import empty_like_unavailable, unavailable_landcover_frame


def test_config_defaults() -> None:
    config = EnvironmentalContextConfig()
    assert config.context_buffer_km == 1.0
    assert config.broad_context_buffer_km == 5.0


def test_config_invalid_buffers_raise() -> None:
    with pytest.raises(ValueError):
        EnvironmentalContextConfig(context_buffer_km=0)
    with pytest.raises(ValueError):
        EnvironmentalContextConfig(context_buffer_km=5.0, broad_context_buffer_km=1.0)


def test_unavailable_landcover_uses_null_not_zero() -> None:
    ids = pd.Series(["E1", "E2"])
    frame = unavailable_landcover_frame(ids)
    assert list(frame["landcover_available"]) == [False, False]
    assert frame["dominant_landcover_fraction"].isna().all()
    assert frame["landcover_class_count"].isna().all()
    assert frame["dominant_landcover_class"].isna().all()


def test_empty_like_unavailable_has_all_flags_false() -> None:
    frame = empty_like_unavailable(pd.Series(["A"]))
    assert frame.loc[0, "landcover_available"] is False or frame.loc[0, "landcover_available"] == False
    assert frame.loc[0, "vegetation_context_available"] == False
    assert frame.loc[0, "satellite_context_available"] == False
    assert np.isnan(frame.loc[0, "satellite_value"])
