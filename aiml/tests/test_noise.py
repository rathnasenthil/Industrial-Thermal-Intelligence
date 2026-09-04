"""Tests for src.event_formation.noise (noise annotation, no deletion)."""

from __future__ import annotations

import pandas as pd

from src.event_formation.config import STDBSCANConfig
from src.event_formation.noise import annotate_noise


def test_noise_rows_are_preserved_with_count_and_reason() -> None:
    noise_df = pd.DataFrame({"latitude": [10.0, 20.0], "longitude": [70.0, 80.0]}, index=[3, 7])
    neighbor_counts = pd.Series({3: 1, 7: 1})
    config = STDBSCANConfig(min_samples=2)

    annotated = annotate_noise(noise_df, neighbor_counts, config)

    assert len(annotated) == 2
    assert "spatiotemporal_neighbor_count" in annotated.columns
    assert "noise_reason" in annotated.columns
    assert list(annotated["spatiotemporal_neighbor_count"]) == [1, 1]
    assert all("isolated" in reason.lower() for reason in annotated["noise_reason"])


def test_noise_reason_mentions_min_samples_when_some_neighbors_exist() -> None:
    noise_df = pd.DataFrame({"latitude": [10.0]}, index=[0])
    neighbor_counts = pd.Series({0: 2})
    config = STDBSCANConfig(min_samples=5)

    annotated = annotate_noise(noise_df, neighbor_counts, config)

    reason = annotated["noise_reason"].iloc[0]
    assert "min_samples=5" in reason
    assert "2" in reason
