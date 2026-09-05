"""Water proximity / coverage context evidence (Stage I.6)."""

from __future__ import annotations

import pandas as pd

from src.environmental_context.config import EnvironmentalContextConfig
from src.environmental_context.vector_presence import compute_vector_presence_context


def compute_water_context(events_df: pd.DataFrame, config: EnvironmentalContextConfig) -> tuple[pd.DataFrame, dict]:
    return compute_vector_presence_context(events_df, config, path=config.water_path, prefix="water")
