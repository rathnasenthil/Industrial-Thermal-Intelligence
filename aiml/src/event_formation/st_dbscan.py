"""
Spatio-Temporal DBSCAN (ST-DBSCAN) clustering for FIRMS thermal detections.

This implements the standard two-parameter ST-DBSCAN formulation
(Birant & Kut, 2007): two detections are "spatio-temporal neighbors" only
if they are within `spatial_eps_km` great-circle distance of each other
AND within `temporal_eps_hours` of each other in `acq_datetime`. Ordinary
density-based clustering (core points / border points / noise) is then
applied using this joint neighborhood definition.

ST-DBSCAN itself is a well-established technique — nothing about the
clustering algorithm here is novel. The engineering contribution of this
module is making it tractable on ~1.17 million detections without ever
materializing an O(n^2) pairwise distance matrix and without holding all
neighbor lists in memory simultaneously:

1. A single global `BallTree` (haversine metric, see `spatial_index.py`)
   indexes every detection's (lat, lon). Radius queries against it cost
   roughly O(log n + k) each, where k is the number of spatial neighbors
   found, instead of O(n) per point.
2. Query points are processed in fixed-size batches (`query_batch_size`),
   so only one batch's worth of neighbor-index arrays is ever held in
   memory at a time — not all ~1.17M lists at once.
3. Each spatial neighbor is additionally checked against
   `temporal_eps_hours` (a cheap vectorized numpy comparison on the small
   per-point neighbor list), producing the final spatio-temporal
   neighbor set for that point (which always includes the point itself,
   at distance 0).
4. Those spatio-temporal neighbor pairs become edges of a sparse graph,
   which is handed to `sklearn.cluster.DBSCAN(metric="precomputed")` —
   scikit-learn's own recommended pattern for large datasets is to
   precompute a sparse radius-neighborhood graph exactly like this,
   rather than let DBSCAN compute a dense distance matrix. This delegates
   the well-tested core/border/noise bookkeeping to scikit-learn instead
   of a hand-rolled Union-Find implementation.

Important note on "boundaries": because step 1 uses a *single* index over
*all* detections, batching in step 2 only changes the order in which
query points are processed — it never changes which points are found as
neighbors of a given point. There is no spatial tiling/partitioning here,
so there is no risk of an event being incorrectly split across a tile
boundary (a real risk with disjoint spatial grid partitioning approaches).

Complexity: building the tree is O(n log n); each of the (n /
query_batch_size) radius-query batches costs roughly O(batch_size *
(log n + k)), so overall neighbor-graph construction is
O(n log n + n * k_avg), where k_avg is the average number of
spatio-temporal neighbors per detection. This is far below the O(n^2)
cost of comparing every pair of ~1.17M detections directly, as long as
k_avg stays modest — see `src.event_formation.benchmark_st_dbscan` for
measurements on this project's real data, including a known
high-density period (crop-residue burning season).

Memory: the sparse neighbor graph's size is O(total edges) = O(n *
k_avg), which is the dominant memory cost after the input arrays
themselves; batching bounds *peak* memory during construction rather than
the final graph size, which is unavoidable no matter the construction
strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.cluster import DBSCAN

from src.event_formation.config import STDBSCANConfig
from src.event_formation.spatial_index import (
    build_haversine_tree,
    km_to_radians,
    to_radians_coords,
)

# Placeholder "distance" written into the sparse precomputed graph for
# every true spatio-temporal-neighbor edge (including self-loops). Its
# absolute value is arbitrary as long as it is < `_PRECOMPUTED_DBSCAN_EPS`
# — DBSCAN(metric="precomputed") only checks "is this pair's stored
# distance <= eps", it does not otherwise use the magnitude.
_EDGE_VALUE = 0.5
_PRECOMPUTED_DBSCAN_EPS = 1.0


@dataclass
class ClusterResult:
    """Result of running ST-DBSCAN over a set of detections.

    Attributes:
        labels: Integer array (len == number of input detections).
            ``-1`` means noise; otherwise a dense, sequential (but
            arbitrary-order) cluster id in ``[0, n_clusters)``.
        neighbor_counts: For each detection, the number of
            spatio-temporal neighbors found (including itself). Useful
            for explaining *why* a point was labeled noise.
        n_clusters: Number of distinct clusters found (excludes noise).
    """

    labels: np.ndarray
    neighbor_counts: np.ndarray
    n_clusters: int


def _epoch_hours(acq_datetime: pd.Series) -> np.ndarray:
    """Convert a tz-aware UTC datetime Series to float hours since epoch."""
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    return ((acq_datetime - epoch) / pd.Timedelta(hours=1)).to_numpy(dtype=np.float64)


def build_spatiotemporal_neighbor_graph(
    latitude: np.ndarray,
    longitude: np.ndarray,
    acq_datetime: pd.Series,
    config: STDBSCANConfig,
) -> tuple[csr_matrix, np.ndarray]:
    """Build the sparse spatio-temporal neighbor graph used by ST-DBSCAN.

    Args:
        latitude: Latitudes in degrees.
        longitude: Longitudes in degrees.
        acq_datetime: Timezone-aware (UTC) acquisition timestamps, same
            length/order as ``latitude``/``longitude``.
        config: Clustering configuration (spatial/temporal epsilon,
            batch size).

    Returns:
        A tuple ``(graph, neighbor_counts)`` where ``graph`` is an
        ``(n, n)`` sparse CSR matrix with an entry for every
        spatio-temporal-neighbor pair (including self-loops on the
        diagonal), suitable for
        ``sklearn.cluster.DBSCAN(metric="precomputed")``, and
        ``neighbor_counts[i]`` is the number of spatio-temporal neighbors
        (including itself) found for detection ``i``.
    """
    n = len(latitude)
    if n == 0:
        return csr_matrix((0, 0)), np.zeros(0, dtype=np.int64)

    tree = build_haversine_tree(latitude, longitude)
    coords_rad = to_radians_coords(latitude, longitude)
    hours = _epoch_hours(acq_datetime)
    radius_rad = km_to_radians(config.spatial_eps_km)

    row_chunks: list[np.ndarray] = []
    col_chunks: list[np.ndarray] = []
    neighbor_counts = np.zeros(n, dtype=np.int64)

    batch_size = max(1, config.query_batch_size)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        neighbor_lists = tree.query_radius(coords_rad[start:end], r=radius_rad)

        for offset, spatial_neighbors in enumerate(neighbor_lists):
            idx = start + offset
            time_delta = np.abs(hours[spatial_neighbors] - hours[idx])
            st_neighbors = spatial_neighbors[time_delta <= config.temporal_eps_hours]

            neighbor_counts[idx] = st_neighbors.size
            row_chunks.append(np.full(st_neighbors.size, idx, dtype=np.int64))
            col_chunks.append(st_neighbors.astype(np.int64, copy=False))

    rows = np.concatenate(row_chunks) if row_chunks else np.empty(0, dtype=np.int64)
    cols = np.concatenate(col_chunks) if col_chunks else np.empty(0, dtype=np.int64)
    values = np.full(rows.shape, _EDGE_VALUE, dtype=np.float64)

    graph = coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()
    return graph, neighbor_counts


def run_st_dbscan(
    latitude: np.ndarray,
    longitude: np.ndarray,
    acq_datetime: pd.Series,
    config: STDBSCANConfig,
) -> ClusterResult:
    """Run ST-DBSCAN over a set of detections.

    Args:
        latitude: Latitudes in degrees.
        longitude: Longitudes in degrees.
        acq_datetime: Timezone-aware (UTC) acquisition timestamps.
        config: Clustering configuration.

    Returns:
        A `ClusterResult`.
    """
    n = len(latitude)
    if n == 0:
        return ClusterResult(labels=np.empty(0, dtype=np.int64), neighbor_counts=np.empty(0, dtype=np.int64), n_clusters=0)

    graph, neighbor_counts = build_spatiotemporal_neighbor_graph(
        latitude, longitude, acq_datetime, config
    )

    dbscan = DBSCAN(
        eps=_PRECOMPUTED_DBSCAN_EPS,
        min_samples=config.min_samples,
        metric="precomputed",
    )
    labels = dbscan.fit_predict(graph)

    n_clusters = int(labels[labels >= 0].max() + 1) if np.any(labels >= 0) else 0
    return ClusterResult(labels=labels.astype(np.int64), neighbor_counts=neighbor_counts, n_clusters=n_clusters)
