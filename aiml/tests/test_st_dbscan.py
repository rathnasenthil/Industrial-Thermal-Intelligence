"""Tests for src.event_formation.st_dbscan (core spatio-temporal clustering)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.event_formation.config import STDBSCANConfig
from src.event_formation.st_dbscan import run_st_dbscan

_BASE_TIME = pd.Timestamp("2023-01-01T06:00:00", tz="UTC")


def _times(hours_offsets: list[float]) -> pd.Series:
    return pd.Series([_BASE_TIME + pd.Timedelta(hours=h) for h in hours_offsets])


def test_nearby_detections_close_in_time_form_one_event() -> None:
    """Scenario 1: several nearby detections close together in time -> one event."""
    lat = np.array([28.60, 28.601, 28.602])
    lon = np.array([77.20, 77.201, 77.202])
    times = _times([0.0, 0.2, 0.4])
    config = STDBSCANConfig(spatial_eps_km=1.5, temporal_eps_hours=36, min_samples=2)

    result = run_st_dbscan(lat, lon, times, config)

    assert (result.labels >= 0).all()
    assert len(set(result.labels)) == 1
    assert result.n_clusters == 1


def test_nearby_but_temporally_separated_detections_form_separate_events() -> None:
    """Scenario 2: nearby in space but time gap exceeds temporal_eps -> separate events."""
    lat = np.array([28.60, 28.601, 28.60, 28.601])
    lon = np.array([77.20, 77.201, 77.20, 77.201])
    # Group A at t=0h/0.2h, group B at t=1000h/1000.2h (>> 36h temporal_eps).
    times = _times([0.0, 0.2, 1000.0, 1000.2])
    config = STDBSCANConfig(spatial_eps_km=1.5, temporal_eps_hours=36, min_samples=2)

    result = run_st_dbscan(lat, lon, times, config)

    assert result.labels[0] == result.labels[1]
    assert result.labels[2] == result.labels[3]
    assert result.labels[0] != result.labels[2]
    assert result.n_clusters == 2


def test_spatially_distant_detections_at_same_time_form_separate_events() -> None:
    """Scenario 3: same time, but far apart in space -> separate events."""
    lat = np.array([28.60, 28.601, 10.00, 10.001])
    lon = np.array([77.20, 77.201, 78.00, 78.001])
    times = _times([0.0, 0.1, 0.0, 0.1])
    config = STDBSCANConfig(spatial_eps_km=1.5, temporal_eps_hours=36, min_samples=2)

    result = run_st_dbscan(lat, lon, times, config)

    assert result.labels[0] == result.labels[1]
    assert result.labels[2] == result.labels[3]
    assert result.labels[0] != result.labels[2]
    assert result.n_clusters == 2


def test_isolated_detection_is_noise() -> None:
    """Scenario 4: a spatially/temporally isolated detection -> noise (-1)."""
    lat = np.array([28.60, 28.601, 5.00])
    lon = np.array([77.20, 77.201, 60.00])
    times = _times([0.0, 0.1, 0.0])
    config = STDBSCANConfig(spatial_eps_km=1.5, temporal_eps_hours=36, min_samples=2)

    result = run_st_dbscan(lat, lon, times, config)

    assert result.labels[2] == -1
    assert result.labels[0] != -1 and result.labels[1] != -1


def test_persistent_chain_of_detections_forms_one_event_via_density_reachability() -> None:
    """Scenario 5: a persistent location re-detected repeatedly, each within
    temporal_eps of its immediate predecessor, must form a single event even
    though the first and last detections are far more than temporal_eps apart
    (this is standard DBSCAN density-reachability/chaining behavior)."""
    n = 6
    lat = np.full(n, 21.50) + np.linspace(0, 0.0005, n)
    lon = np.full(n, 82.10) + np.linspace(0, 0.0005, n)
    # Consecutive detections 20h apart (within temporal_eps=36h of each
    # neighbor), spanning 100h total (>> temporal_eps) end-to-end.
    times = _times([i * 20.0 for i in range(n)])
    config = STDBSCANConfig(spatial_eps_km=1.5, temporal_eps_hours=36, min_samples=2)

    result = run_st_dbscan(lat, lon, times, config)

    assert (result.labels == result.labels[0]).all()
    assert result.labels[0] != -1


def test_min_samples_threshold_is_respected() -> None:
    """A pair of mutually-close detections should be noise if min_samples
    requires more corroboration than is available."""
    lat = np.array([28.60, 28.601])
    lon = np.array([77.20, 77.201])
    times = _times([0.0, 0.1])
    config = STDBSCANConfig(spatial_eps_km=1.5, temporal_eps_hours=36, min_samples=3)

    result = run_st_dbscan(lat, lon, times, config)

    assert (result.labels == -1).all()


@pytest.mark.parametrize("batch_size", [1, 2, 5, 1000])
def test_query_batch_size_does_not_change_clustering_result(batch_size: int) -> None:
    """Batching only affects processing order via a single global spatial
    index, never which points are neighbors of each other — so results must
    be identical regardless of batch size (no tile-boundary splitting)."""
    rng = np.random.default_rng(42)
    n = 30
    # Three well-separated clumps, so we can check clumps stay intact
    # regardless of how query batches slice through them.
    clumps_lat = []
    clumps_lon = []
    for center_lat, center_lon in [(28.60, 77.20), (13.00, 80.20), (23.00, 72.60)]:
        clumps_lat.append(center_lat + rng.normal(0, 0.001, n // 3))
        clumps_lon.append(center_lon + rng.normal(0, 0.001, n // 3))
    lat = np.concatenate(clumps_lat)
    lon = np.concatenate(clumps_lon)
    times = _times([float(i % 5) for i in range(len(lat))])

    config_small_batch = STDBSCANConfig(spatial_eps_km=1.5, temporal_eps_hours=36, min_samples=2, query_batch_size=batch_size)
    config_large_batch = STDBSCANConfig(spatial_eps_km=1.5, temporal_eps_hours=36, min_samples=2, query_batch_size=100_000)

    result_small = run_st_dbscan(lat, lon, times, config_small_batch)
    result_large = run_st_dbscan(lat, lon, times, config_large_batch)

    # Same partition into groups (up to arbitrary cluster-id numbering):
    # verify pairwise "same cluster" relationships are identical.
    same_small = result_small.labels[:, None] == result_small.labels[None, :]
    same_large = result_large.labels[:, None] == result_large.labels[None, :]
    # Noise points (-1) trivially satisfy label equality with other noise
    # points under this check, so also require both sides agree on noise.
    noise_small = result_small.labels == -1
    noise_large = result_large.labels == -1
    np.testing.assert_array_equal(noise_small, noise_large)
    np.testing.assert_array_equal(same_small & ~noise_small[:, None], same_large & ~noise_large[:, None])


def test_neighbor_counts_include_self() -> None:
    lat = np.array([28.60, 28.601, 28.602])
    lon = np.array([77.20, 77.201, 77.202])
    times = _times([0.0, 0.1, 0.2])
    config = STDBSCANConfig(spatial_eps_km=1.5, temporal_eps_hours=36, min_samples=2)

    result = run_st_dbscan(lat, lon, times, config)

    assert (result.neighbor_counts >= 1).all()
    assert result.neighbor_counts[0] == 3


def test_empty_input_returns_empty_result() -> None:
    config = STDBSCANConfig()
    result = run_st_dbscan(np.array([]), np.array([]), pd.Series([], dtype="datetime64[ns, UTC]"), config)
    assert len(result.labels) == 0
    assert result.n_clusters == 0
